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

#[cfg(feature = "jit")]
unsafe impl Send for JitSimd {}
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
        flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;
        flag_builder.set("use_colocated_libcalls", "false").map_err(|e| e.to_string())?;

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
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Scale,
                            params: inst.params,
                        });

                        let s_s = builder.ins().f32const(inst.params[0]);
                        let s = builder.ins().splat(simd_type, s_s);

                        // p = p / s
                        let new_x0 = builder.ins().fdiv(curr_x.0, s);
                        let new_x1 = builder.ins().fdiv(curr_x.1, s);
                        let new_y0 = builder.ins().fdiv(curr_y.0, s);
                        let new_y1 = builder.ins().fdiv(curr_y.1, s);
                        let new_z0 = builder.ins().fdiv(curr_z.0, s);
                        let new_z1 = builder.ins().fdiv(curr_z.1, s);

                        curr_x = (new_x0, new_x1);
                        curr_y = (new_y0, new_y1);
                        curr_z = (new_z0, new_z1);

                        // scale *= s
                        let new_scale0 = builder.ins().fmul(curr_scale.0, s);
                        let new_scale1 = builder.ins().fmul(curr_scale.1, s);
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
        module
            .finalize_definitions()
            .map_err(|e| e.to_string())?;

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
    pub unsafe fn eval(
        &self,
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        pout: *mut f32,
    ) {
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
        let shape = SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0));
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
