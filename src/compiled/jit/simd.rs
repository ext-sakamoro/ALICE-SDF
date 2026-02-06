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
type SimdSdfFn = unsafe extern "C" fn(
    px: *const f32,
    py: *const f32,
    pz: *const f32,
    out: *mut f32,
);

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
    param: f32,
    /// Flag indicating this transform was folded (no-op)
    folded: bool,
}

/// Epsilon for constant folding
const FOLD_EPSILON: f32 = 1e-6;

impl JitSimdSdf {
    /// Compile SDF to native SIMD machine code with Aggressive optimizations
    pub fn compile(sdf: &CompiledSdf) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;

        // Use colocated libcalls to avoid PLT overhead
        flag_builder.set("use_colocated_libcalls", "true").map_err(|e| e.to_string())?;

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
            .declare_function("eval_sdf_simd_deep_fried", Linkage::Export, &ctx.func.signature)
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

                    OpCode::Union => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack.push((
                            builder.ins().fmin(a.0, b.0),
                            builder.ins().fmin(a.1, b.1)
                        ));
                    }

                    OpCode::Intersection => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack.push((
                            builder.ins().fmax(a.0, b.0),
                            builder.ins().fmax(a.1, b.1)
                        ));
                    }

                    OpCode::Subtraction => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let neg_b0 = builder.ins().fneg(b.0);
                        let neg_b1 = builder.ins().fneg(b.1);
                        value_stack.push((
                            builder.ins().fmax(a.0, neg_b0),
                            builder.ins().fmax(a.1, neg_b1)
                        ));
                    }

                    // Division Exorcism: Smooth Ops use mul(1/k) instead of div(k)
                    OpCode::SmoothUnion => {
                        let k = inst.params[0];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        if k.abs() < FOLD_EPSILON {
                            value_stack.push((
                                builder.ins().fmin(a.0, b.0),
                                builder.ins().fmin(a.1, b.1)
                            ));
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
                            value_stack.push((
                                builder.ins().fmax(a.0, b.0),
                                builder.ins().fmax(a.1, b.1)
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
                                builder.ins().fmax(a.1, neg_b1)
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

                    OpCode::Translate => {
                        let tx = inst.params[0];
                        let ty = inst.params[1];
                        let tz = inst.params[2];

                        if tx.abs() < FOLD_EPSILON && ty.abs() < FOLD_EPSILON && tz.abs() < FOLD_EPSILON {
                            coord_stack.push(SimdCoordState {
                                x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                                opcode: OpCode::Translate, param: 0.0, folded: true,
                            });
                            continue;
                        }

                        coord_stack.push(SimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Translate, param: 0.0, folded: false,
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
                        let s = inst.params[0];

                        if (s - 1.0).abs() < FOLD_EPSILON {
                            coord_stack.push(SimdCoordState {
                                x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                                opcode: OpCode::Scale, param: 0.0, folded: true,
                            });
                            continue;
                        }

                        coord_stack.push(SimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Scale, param: 0.0, folded: false,
                        });

                        // Pre-compute reciprocal
                        let inv_s = 1.0 / s;
                        let inv_s_s = builder.ins().f32const(inv_s);
                        let inv_s_v = builder.ins().splat(vec_type, inv_s_s);

                        let s_s = builder.ins().f32const(s);
                        let s_v = builder.ins().splat(vec_type, s_s);

                        // Multiply by reciprocal instead of divide
                        let nx0 = builder.ins().fmul(curr_x.0, inv_s_v);
                        let ny0 = builder.ins().fmul(curr_y.0, inv_s_v);
                        let nz0 = builder.ins().fmul(curr_z.0, inv_s_v);
                        let nx1 = builder.ins().fmul(curr_x.1, inv_s_v);
                        let ny1 = builder.ins().fmul(curr_y.1, inv_s_v);
                        let nz1 = builder.ins().fmul(curr_z.1, inv_s_v);

                        let ns0 = builder.ins().fmul(curr_scale.0, s_v);
                        let ns1 = builder.ins().fmul(curr_scale.1, s_v);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                        curr_scale = (ns0, ns1);
                    }

                    OpCode::Round => {
                        let r = inst.params[0];
                        coord_stack.push(SimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Round, param: r, folded: r.abs() < FOLD_EPSILON
                        });
                    }

                    OpCode::Onion => {
                        let t = inst.params[0];
                        coord_stack.push(SimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Onion, param: t, folded: t.abs() < FOLD_EPSILON
                        });
                    }

                    OpCode::PopTransform => {
                        if let Some(state) = coord_stack.pop() {
                            if !state.folded {
                                match state.opcode {
                                    OpCode::Round => {
                                        let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                                        let r_s = builder.ins().f32const(state.param);
                                        let r = builder.ins().splat(vec_type, r_s);
                                        value_stack.push((
                                            builder.ins().fsub(d.0, r),
                                            builder.ins().fsub(d.1, r)
                                        ));
                                    }
                                    OpCode::Onion => {
                                        let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                                        let t_s = builder.ins().f32const(state.param);
                                        let t = builder.ins().splat(vec_type, t_s);
                                        let a0 = builder.ins().fabs(d.0);
                                        let a1 = builder.ins().fabs(d.1);
                                        value_stack.push((
                                            builder.ins().fsub(a0, t),
                                            builder.ins().fsub(a1, t)
                                        ));
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

        module.define_function(func_id, &mut ctx).map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = module.get_finalized_function(func_id);

        Ok(JitSimdSdf { module, func_ptr: code })
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
                    self.eval_8_raw(
                        x_ptr.add(i),
                        y_ptr.add(i),
                        z_ptr.add(i),
                        out_ptr.add(i)
                    );
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
                std::ptr::copy_nonoverlapping(x.as_ptr().add(offset), x_pad.as_mut_ptr(), remainder);
                std::ptr::copy_nonoverlapping(y.as_ptr().add(offset), y_pad.as_mut_ptr(), remainder);
                std::ptr::copy_nonoverlapping(z.as_ptr().add(offset), z_pad.as_mut_ptr(), remainder);

                self.eval_8_raw(x_pad.as_ptr(), y_pad.as_ptr(), z_pad.as_ptr(), out_pad.as_mut_ptr());

                std::ptr::copy_nonoverlapping(out_pad.as_ptr(), results.as_mut_ptr().add(offset), remainder);
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
        Self { params_ptr, param_index: 0, params: Vec::new() }
    }

    /// Emit a parameter: load scalar from params_ptr buffer
    fn emit(&mut self, builder: &mut FunctionBuilder, value: f32) -> Value {
        self.params.push(value);
        let idx = self.param_index;
        self.param_index += 1;
        let mut flags = MemFlags::new();
        flags.set_notrap();
        builder.ins().load(types::F32, flags, self.params_ptr, (idx * 4) as i32)
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
    /// Pre-splatted parameter vector for Round/Onion
    param_vec: Value,
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
        flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;
        flag_builder.set("use_colocated_libcalls", "true").map_err(|e| e.to_string())?;
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
            .declare_function("eval_sdf_simd_dynamic", Linkage::Export, &ctx.func.signature)
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

                    OpCode::Union => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack.push((
                            builder.ins().fmin(a.0, b.0),
                            builder.ins().fmin(a.1, b.1),
                        ));
                    }

                    OpCode::Intersection => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack.push((
                            builder.ins().fmax(a.0, b.0),
                            builder.ins().fmax(a.1, b.1),
                        ));
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
                        let inv_k_raw = if k_raw.abs() < 1e-10 { 1.0 } else { 1.0 / k_raw };
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
                        let inv_k_raw = if k_raw.abs() < 1e-10 { 1.0 } else { 1.0 / k_raw };
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
                        let inv_k_raw = if k_raw.abs() < 1e-10 { 1.0 } else { 1.0 / k_raw };
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

                    OpCode::Translate => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Translate, param_vec: zero_vec,
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

                    // Division Exorcism: Scale uses mul(inv_s)
                    OpCode::Scale => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Scale, param_vec: zero_vec,
                        });

                        let s_raw = inst.params[0];
                        let inv_s_raw = if s_raw.abs() < 1e-10 { 1.0 } else { 1.0 / s_raw };

                        let s_v = emitter.emit_splat(&mut builder, s_raw);
                        let inv_s_v = emitter.emit_splat(&mut builder, inv_s_raw);

                        let nx0 = builder.ins().fmul(curr_x.0, inv_s_v);
                        let ny0 = builder.ins().fmul(curr_y.0, inv_s_v);
                        let nz0 = builder.ins().fmul(curr_z.0, inv_s_v);
                        let nx1 = builder.ins().fmul(curr_x.1, inv_s_v);
                        let ny1 = builder.ins().fmul(curr_y.1, inv_s_v);
                        let nz1 = builder.ins().fmul(curr_z.1, inv_s_v);

                        let ns0 = builder.ins().fmul(curr_scale.0, s_v);
                        let ns1 = builder.ins().fmul(curr_scale.1, s_v);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                        curr_scale = (ns0, ns1);
                    }

                    OpCode::Round => {
                        let r_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Round, param_vec: r_v,
                        });
                    }

                    OpCode::Onion => {
                        let t_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x, y: curr_y, z: curr_z, scale: curr_scale,
                            opcode: OpCode::Onion, param_vec: t_v,
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

        module.define_function(func_id, &mut ctx).map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = module.get_finalized_function(func_id);

        Ok(JitSimdSdfDynamic { module, func_ptr: code, params: initial_params })
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
                    self.eval_8_raw(
                        x_ptr.add(i), y_ptr.add(i),
                        z_ptr.add(i), out_ptr.add(i),
                    );
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
                std::ptr::copy_nonoverlapping(x.as_ptr().add(offset), x_pad.as_mut_ptr(), remainder);
                std::ptr::copy_nonoverlapping(y.as_ptr().add(offset), y_pad.as_mut_ptr(), remainder);
                std::ptr::copy_nonoverlapping(z.as_ptr().add(offset), z_pad.as_mut_ptr(), remainder);
                self.eval_8_raw(x_pad.as_ptr(), y_pad.as_ptr(), z_pad.as_ptr(), out_pad.as_mut_ptr());
                std::ptr::copy_nonoverlapping(out_pad.as_ptr(), results.as_mut_ptr().add(offset), remainder);
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
            OpCode::SmoothUnion | OpCode::SmoothIntersection | OpCode::SmoothSubtraction => {
                let k = inst.params[0];
                let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / k };
                params.push(k);
                params.push(inv_k);
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
            OpCode::Round => {
                params.push(inst.params[0]);
            }
            OpCode::Onion => {
                params.push(inst.params[0]);
            }
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
                i, results[i], expected
            );
        }
    }

    #[test]
    fn test_jit_simd_box() {
        let box3d = SdfNode::Box3d { half_extents: Vec3::new(1.0, 0.5, 0.5) };
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
                i, results[i], expected
            );
        }
    }

    #[test]
    fn test_jit_simd_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.union(
            SdfNode::Box3d { half_extents: Vec3::splat(0.5) }.translate(2.0, 0.0, 0.0),
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
                i, results[i], expected
            );
        }
    }

    #[test]
    fn test_jit_simd_smooth_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Box3d { half_extents: Vec3::splat(0.5) }.translate(1.5, 0.0, 0.0),
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
                i, results[i], expected
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
                i, results[i], expected
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
                i, results[i], expected
            );
        }
    }

    #[test]
    fn test_jit_simd_soa() {
        use crate::soa::SoAPoints;

        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Cylinder { radius: 0.3, half_height: 1.5 },
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
                i, results[i], expected
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
                i, results[i], expected
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
                i, results[i], expected
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
        assert!(approx_eq(results1[0], expected1, 0.001),
            "Before update: jit={}, cpu={}", results1[0], expected1);

        // Update to radius=2.0 (no recompilation!)
        let sphere2 = SdfNode::Sphere { radius: 2.0 };
        let compiled2 = CompiledSdf::compile(&sphere2);
        jit.update_params(&compiled2);

        let results2 = unsafe { jit.eval_8(&x, &y, &z) };
        let expected2 = eval(&sphere2, Vec3::new(1.5, 0.0, 0.0));
        assert!(approx_eq(results2[0], expected2, 0.001),
            "After update: jit={}, cpu={}", results2[0], expected2);
    }

    #[test]
    fn test_jit_simd_dynamic_smooth_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Box3d { half_extents: Vec3::splat(0.5) }.translate(1.5, 0.0, 0.0),
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
                i, results[i], expected
            );
        }
    }

    #[test]
    fn test_jit_simd_dynamic_extract_params() {
        let shape = SdfNode::Sphere { radius: 1.5 }
            .translate(2.0, 3.0, 4.0);
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
                i, results[i], expected
            );
        }
    }
}
