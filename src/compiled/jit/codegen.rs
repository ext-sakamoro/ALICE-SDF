//! JIT Code Generation: SDF to Cranelift IR (Deep Fried v2)
//!
//! Optimizations over v1:
//! - **Division Exorcism**: All constant divisions → mul(reciprocal)
//! - **FMA**: Fused Multiply-Add for length/dot helpers
//! - **Dynamic Parameters**: Runtime parameter updates without recompilation
//!
//! Author: Moroya Sakamoto

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, UserFuncName, Value};
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};

use crate::types::SdfNode;
use super::runtime::JitError;

// ============ Parameter Emitter ============

/// Parameter source for JIT code generation.
///
/// In hardcoded mode, emits `f32const` instructions (constants baked into code).
/// In dynamic mode, emits `load` instructions from a parameter pointer.
struct JitParamEmitter {
    params_ptr: Option<Value>,
    param_index: usize,
    params: Vec<f32>,
}

impl JitParamEmitter {
    fn hardcoded() -> Self {
        Self { params_ptr: None, param_index: 0, params: Vec::new() }
    }

    fn dynamic(params_ptr: Value) -> Self {
        Self { params_ptr: Some(params_ptr), param_index: 0, params: Vec::new() }
    }

    /// Emit a shape parameter value.
    /// Hardcoded: f32const (baked). Dynamic: load from params_ptr.
    fn emit(&mut self, builder: &mut FunctionBuilder, value: f32) -> Value {
        self.params.push(value);
        let idx = self.param_index;
        self.param_index += 1;
        match self.params_ptr {
            None => builder.ins().f32const(value),
            Some(ptr) => {
                let mut flags = MemFlags::new();
                flags.set_notrap();
                builder.ins().load(types::F32, flags, ptr, (idx * 4) as i32)
            }
        }
    }

    fn into_params(self) -> Vec<f32> {
        self.params
    }
}

// ============ JIT Compiler ============

/// JIT Compiler for SDF evaluation (Deep Fried v2)
pub struct JitCompiler<'a> {
    module: &'a mut JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl<'a> JitCompiler<'a> {
    pub fn new(module: &'a mut JITModule) -> Self {
        let ctx = module.make_context();
        let func_ctx = FunctionBuilderContext::new();
        JitCompiler { module, ctx, func_ctx }
    }

    /// Compile an SDF node to a function: fn(f32, f32, f32) -> f32
    pub fn compile_sdf(&mut self, node: &SdfNode) -> Result<FuncId, JitError> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32)); // x
        sig.params.push(AbiParam::new(types::F32)); // y
        sig.params.push(AbiParam::new(types::F32)); // z
        sig.returns.push(AbiParam::new(types::F32)); // distance

        let func_id = self
            .module
            .declare_function("sdf_eval", Linkage::Export, &sig)
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        self.ctx.func.signature = sig;
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let x = builder.block_params(entry_block)[0];
            let y = builder.block_params(entry_block)[1];
            let z = builder.block_params(entry_block)[2];

            let mut emitter = JitParamEmitter::hardcoded();
            let result = compile_node(&mut builder, node, x, y, z, &mut emitter)?;
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JitError::CompilationError(e.to_string()))?;
        self.module.clear_context(&mut self.ctx);
        Ok(func_id)
    }

    /// Compile an SDF node with dynamic parameters: fn(f32, f32, f32, *const f32) -> f32
    ///
    /// Returns the function ID and the initial parameter values.
    pub fn compile_sdf_dynamic(&mut self, node: &SdfNode) -> Result<(FuncId, Vec<f32>), JitError> {
        let ptr_type = self.module.target_config().pointer_type();

        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32));    // x
        sig.params.push(AbiParam::new(types::F32));    // y
        sig.params.push(AbiParam::new(types::F32));    // z
        sig.params.push(AbiParam::new(ptr_type));       // params_ptr
        sig.returns.push(AbiParam::new(types::F32));    // distance

        let func_id = self
            .module
            .declare_function("sdf_eval_dynamic", Linkage::Export, &sig)
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        self.ctx.func.signature = sig;
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        let params;
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let x = builder.block_params(entry_block)[0];
            let y = builder.block_params(entry_block)[1];
            let z = builder.block_params(entry_block)[2];
            let params_ptr = builder.block_params(entry_block)[3];

            let mut emitter = JitParamEmitter::dynamic(params_ptr);
            let result = compile_node(&mut builder, node, x, y, z, &mut emitter)?;
            builder.ins().return_(&[result]);
            builder.finalize();

            params = emitter.into_params();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JitError::CompilationError(e.to_string()))?;
        self.module.clear_context(&mut self.ctx);
        Ok((func_id, params))
    }
}

// ============ Node Compilation ============

/// Compile a single SDF node, returning the IR value for the distance
fn compile_node(
    builder: &mut FunctionBuilder,
    node: &SdfNode,
    x: Value,
    y: Value,
    z: Value,
    emitter: &mut JitParamEmitter,
) -> Result<Value, JitError> {
    match node {
        // ============ Primitives ============

        SdfNode::Sphere { radius } => {
            let r = emitter.emit(builder, *radius);
            let len = emit_length3(builder, x, y, z);
            Ok(builder.ins().fsub(len, r))
        }

        SdfNode::Box3d { half_extents } => {
            let hx = emitter.emit(builder, half_extents.x);
            let hy = emitter.emit(builder, half_extents.y);
            let hz = emitter.emit(builder, half_extents.z);

            let ax = builder.ins().fabs(x);
            let ay = builder.ins().fabs(y);
            let az = builder.ins().fabs(z);

            let qx = builder.ins().fsub(ax, hx);
            let qy = builder.ins().fsub(ay, hy);
            let qz = builder.ins().fsub(az, hz);

            let zero = builder.ins().f32const(0.0);
            let mx = builder.ins().fmax(qx, zero);
            let my = builder.ins().fmax(qy, zero);
            let mz = builder.ins().fmax(qz, zero);

            let outside = emit_length3(builder, mx, my, mz);

            let inner_max = builder.ins().fmax(qy, qz);
            let outer_max = builder.ins().fmax(qx, inner_max);
            let inside = builder.ins().fmin(outer_max, zero);

            Ok(builder.ins().fadd(outside, inside))
        }

        SdfNode::Cylinder { radius, half_height } => {
            let r = emitter.emit(builder, *radius);
            let h = emitter.emit(builder, *half_height);
            let zero = builder.ins().f32const(0.0);

            let len_xz = emit_length2(builder, x, z);
            let dx = builder.ins().fsub(len_xz, r);

            let ay = builder.ins().fabs(y);
            let dy = builder.ins().fsub(ay, h);

            let inner = builder.ins().fmax(dx, dy);
            let inside = builder.ins().fmin(inner, zero);

            let mx = builder.ins().fmax(dx, zero);
            let my = builder.ins().fmax(dy, zero);
            let outside = emit_length2(builder, mx, my);

            Ok(builder.ins().fadd(inside, outside))
        }

        SdfNode::Torus { major_radius, minor_radius } => {
            let major = emitter.emit(builder, *major_radius);
            let minor = emitter.emit(builder, *minor_radius);

            let len_xz = emit_length2(builder, x, z);
            let qx = builder.ins().fsub(len_xz, major);
            let len_q = emit_length2(builder, qx, y);
            Ok(builder.ins().fsub(len_q, minor))
        }

        SdfNode::Plane { normal, distance } => {
            let nx = emitter.emit(builder, normal.x);
            let ny = emitter.emit(builder, normal.y);
            let nz = emitter.emit(builder, normal.z);
            let dist = emitter.emit(builder, *distance);

            let dot = emit_dot3(builder, x, y, z, nx, ny, nz);
            Ok(builder.ins().fadd(dot, dist))
        }

        SdfNode::Capsule { point_a, point_b, radius } => {
            // Emit shape params
            let ax_v = emitter.emit(builder, point_a.x);
            let ay_v = emitter.emit(builder, point_a.y);
            let az_v = emitter.emit(builder, point_a.z);
            let bx_v = emitter.emit(builder, point_b.x);
            let by_v = emitter.emit(builder, point_b.y);
            let bz_v = emitter.emit(builder, point_b.z);
            let r = emitter.emit(builder, *radius);

            // Division Exorcism: pre-compute 1/dot(ba,ba)
            let ba = *point_b - *point_a;
            let dot_ba_ba = ba.dot(ba);
            let inv_dbb = if dot_ba_ba.abs() < 1e-10 { 1.0 } else { 1.0 / dot_ba_ba };
            let inv_dbb_v = emitter.emit(builder, inv_dbb);

            // pa = p - a
            let pax = builder.ins().fsub(x, ax_v);
            let pay = builder.ins().fsub(y, ay_v);
            let paz = builder.ins().fsub(z, az_v);

            // ba = b - a
            let bax = builder.ins().fsub(bx_v, ax_v);
            let bay = builder.ins().fsub(by_v, ay_v);
            let baz = builder.ins().fsub(bz_v, az_v);

            // h = clamp(dot(pa, ba) * inv_dot_ba_ba, 0, 1)
            let dot_pa_ba = emit_dot3(builder, pax, pay, paz, bax, bay, baz);
            let h_raw = builder.ins().fmul(dot_pa_ba, inv_dbb_v); // Division Exorcism!
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let h_clamped = builder.ins().fmin(h_raw, one);
            let h = builder.ins().fmax(zero, h_clamped);

            // pa - ba * h
            let bhx = builder.ins().fmul(bax, h);
            let bhy = builder.ins().fmul(bay, h);
            let bhz = builder.ins().fmul(baz, h);
            let dx = builder.ins().fsub(pax, bhx);
            let dy = builder.ins().fsub(pay, bhy);
            let dz = builder.ins().fsub(paz, bhz);

            let len = emit_length3(builder, dx, dy, dz);
            Ok(builder.ins().fsub(len, r))
        }

        SdfNode::Cone { radius, half_height } => {
            let r = emitter.emit(builder, *radius);
            let h = emitter.emit(builder, *half_height);
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let neg_one = builder.ins().f32const(-1.0);
            let two = builder.ins().f32const(2.0);

            // q_x = length(p.xz)
            let q_x = emit_length2(builder, x, z);
            let q_y = y;

            // k2 = (-radius, 2*h)
            let k2x = builder.ins().fneg(r);
            let k2y = builder.ins().fmul(two, h);
            // Division Exorcism: pre-compute 1/k2_dot
            let inv_k2_dot = {
                let k2d_val = (*radius) * (*radius) + (2.0 * *half_height) * (2.0 * *half_height);
                let inv = if k2d_val.abs() < 1e-10 { 1.0 } else { 1.0 / k2d_val };
                emitter.emit(builder, inv)
            };

            // ca_r = q_y < 0 ? radius : 0 (branchless select)
            let cmp_neg = builder.ins().fcmp(FloatCC::LessThan, q_y, zero);
            let ca_r = builder.ins().select(cmp_neg, r, zero);

            let min_q_ca = builder.ins().fmin(q_x, ca_r);
            let ca_x = builder.ins().fsub(q_x, min_q_ca);
            let abs_qy = builder.ins().fabs(q_y);
            let ca_y = builder.ins().fsub(abs_qy, h);

            // t = clamp(dot(-q_x, h-q_y, k2) * inv_k2_dot, 0, 1)
            let neg_qx = builder.ins().fneg(q_x);
            let diff_y = builder.ins().fsub(h, q_y);
            let dy_k2y = builder.ins().fmul(diff_y, k2y);
            let dot_val = builder.ins().fma(neg_qx, k2x, dy_k2y);
            let t_raw = builder.ins().fmul(dot_val, inv_k2_dot);
            let t_min = builder.ins().fmin(t_raw, one);
            let t = builder.ins().fmax(zero, t_min);

            // cb = (q_x + k2x*t, q_y - h + k2y*t)
            let cb_x = builder.ins().fma(k2x, t, q_x);
            let qy_h = builder.ins().fsub(q_y, h);
            let cb_y = builder.ins().fma(k2y, t, qy_h);

            // s = (cb_x < 0 && ca_y < 0) ? -1 : 1
            let cmp_cbx = builder.ins().fcmp(FloatCC::LessThan, cb_x, zero);
            let cmp_cay = builder.ins().fcmp(FloatCC::LessThan, ca_y, zero);
            let both = builder.ins().band(cmp_cbx, cmp_cay);
            let s = builder.ins().select(both, neg_one, one);

            // d2 = min(ca_x²+ca_y², cb_x²+cb_y²)
            let ca_yy = builder.ins().fmul(ca_y, ca_y);
            let ca_sq = builder.ins().fma(ca_x, ca_x, ca_yy);
            let cb_yy = builder.ins().fmul(cb_y, cb_y);
            let cb_sq = builder.ins().fma(cb_x, cb_x, cb_yy);
            let d2 = builder.ins().fmin(ca_sq, cb_sq);

            let d = builder.ins().sqrt(d2);
            Ok(builder.ins().fmul(s, d))
        }

        SdfNode::Ellipsoid { radii } => {
            // Division Exorcism: pre-compute reciprocals
            let inv_rx = emitter.emit(builder, 1.0 / radii.x);
            let inv_ry = emitter.emit(builder, 1.0 / radii.y);
            let inv_rz = emitter.emit(builder, 1.0 / radii.z);
            let inv_rx2 = emitter.emit(builder, 1.0 / (radii.x * radii.x));
            let inv_ry2 = emitter.emit(builder, 1.0 / (radii.y * radii.y));
            let inv_rz2 = emitter.emit(builder, 1.0 / (radii.z * radii.z));

            let px = builder.ins().fmul(x, inv_rx);
            let py = builder.ins().fmul(y, inv_ry);
            let pz = builder.ins().fmul(z, inv_rz);
            let k0 = emit_length3(builder, px, py, pz);

            let qx = builder.ins().fmul(x, inv_rx2);
            let qy = builder.ins().fmul(y, inv_ry2);
            let qz = builder.ins().fmul(z, inv_rz2);
            let k1 = emit_length3(builder, qx, qy, qz);

            let eps = builder.ins().f32const(1e-10);
            let k1_safe = builder.ins().fadd(k1, eps);

            let one = builder.ins().f32const(1.0);
            let k0_minus_1 = builder.ins().fsub(k0, one);
            let num = builder.ins().fmul(k0, k0_minus_1);
            Ok(builder.ins().fdiv(num, k1_safe)) // data-dependent, can't precompute
        }

        SdfNode::RoundedCone { r1, r2, half_height } => {
            let r1_val = emitter.emit(builder, *r1);
            let r2_val = emitter.emit(builder, *r2);
            let hh = emitter.emit(builder, *half_height);
            let zero = builder.ins().f32const(0.0);

            // Pre-compute constants (Division Exorcism)
            let h_scalar = half_height * 2.0;
            let inv_h = if h_scalar.abs() < 1e-10 { 1.0 } else { 1.0 / h_scalar };
            let b_scalar = (r1 - r2) * inv_h;
            let a_scalar = (1.0 - b_scalar * b_scalar).max(0.0).sqrt();
            let ah_scalar = a_scalar * h_scalar;

            let h = emitter.emit(builder, h_scalar);
            let b = emitter.emit(builder, b_scalar);
            let a = emitter.emit(builder, a_scalar);
            let ah = emitter.emit(builder, ah_scalar);

            // q_x = length(p.x, p.z), q_y = p.y + half_height
            let q_x = emit_length2(builder, x, z);
            let q_y = builder.ins().fadd(y, hh);

            // k = q_x * (-b) + q_y * a
            let neg_b = builder.ins().fneg(b);
            let qy_a = builder.ins().fmul(q_y, a);
            let k = builder.ins().fma(q_x, neg_b, qy_a);

            // Case 1 (k < 0): length(q_x, q_y) - r1
            let len1 = emit_length2(builder, q_x, q_y);
            let d1 = builder.ins().fsub(len1, r1_val);

            // Case 2 (k > a*h): length(q_x, q_y - h) - r2
            let qy_h = builder.ins().fsub(q_y, h);
            let len2 = emit_length2(builder, q_x, qy_h);
            let d2 = builder.ins().fsub(len2, r2_val);

            // Case 3: q_x * a + q_y * b - r1
            let neg_r1 = builder.ins().fneg(r1_val);
            let qy_b_nr1 = builder.ins().fma(q_y, b, neg_r1);
            let d3 = builder.ins().fma(q_x, a, qy_b_nr1);

            // Branchless select
            let cmp_neg = builder.ins().fcmp(FloatCC::LessThan, k, zero);
            let cmp_gt = builder.ins().fcmp(FloatCC::GreaterThan, k, ah);
            let d_inner = builder.ins().select(cmp_gt, d2, d3);
            Ok(builder.ins().select(cmp_neg, d1, d_inner))
        }

        SdfNode::Pyramid { half_height } => {
            // Pre-compute constants (Division Exorcism)
            let h_scalar = half_height * 2.0;
            let m2_scalar = h_scalar * h_scalar + 0.25;
            let inv_m2_scalar = 1.0 / m2_scalar;
            let inv_m2_025_scalar = 1.0 / (m2_scalar + 0.25);

            let hh = emitter.emit(builder, *half_height);
            let h = emitter.emit(builder, h_scalar);
            let m2 = emitter.emit(builder, m2_scalar);
            let inv_m2 = emitter.emit(builder, inv_m2_scalar);
            let inv_m2_025 = emitter.emit(builder, inv_m2_025_scalar);

            let half = builder.ins().f32const(0.5);
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let neg_one = builder.ins().f32const(-1.0);

            // py = p.y + half_height
            let py = builder.ins().fadd(y, hh);

            // Branchless swap: px = max(|p.x|, |p.z|), pz = min(|p.x|, |p.z|)
            let abs_px = builder.ins().fabs(x);
            let abs_pz = builder.ins().fabs(z);
            let px_s = builder.ins().fmax(abs_px, abs_pz);
            let pz_s = builder.ins().fmin(abs_px, abs_pz);

            // px -= 0.5, pz -= 0.5
            let px_adj = builder.ins().fsub(px_s, half);
            let pz_adj = builder.ins().fsub(pz_s, half);

            // q = (pz_adj, h*py - 0.5*px_adj, h*px_adj + 0.5*py)
            let qx = pz_adj;
            let neg_half = builder.ins().fneg(half);
            let nhalf_px = builder.ins().fmul(neg_half, px_adj);
            let qy = builder.ins().fma(h, py, nhalf_px);
            let half_py = builder.ins().fmul(half, py);
            let qz = builder.ins().fma(h, px_adj, half_py);

            // s = max(-qx, 0)
            let neg_qx = builder.ins().fneg(qx);
            let s = builder.ins().fmax(neg_qx, zero);

            // t = clamp((qy - 0.5*pz_adj) * inv_m2_025, 0, 1)  — Division Exorcism
            let half_pz = builder.ins().fmul(half, pz_adj);
            let qy_sub = builder.ins().fsub(qy, half_pz);
            let t_raw = builder.ins().fmul(qy_sub, inv_m2_025);
            let t_min = builder.ins().fmin(t_raw, one);
            let t = builder.ins().fmax(zero, t_min);

            // a = m2 * (qx + s)² + qy²
            let qx_s = builder.ins().fadd(qx, s);
            let qx_s_sq = builder.ins().fmul(qx_s, qx_s);
            let qy_sq = builder.ins().fmul(qy, qy);
            let a_val = builder.ins().fma(m2, qx_s_sq, qy_sq);

            // b = m2 * (qx + 0.5*t)² + (qy - m2*t)²
            let half_t = builder.ins().fmul(half, t);
            let qx_ht = builder.ins().fadd(qx, half_t);
            let qx_ht_sq = builder.ins().fmul(qx_ht, qx_ht);
            let m2_t = builder.ins().fmul(m2, t);
            let qy_m2t = builder.ins().fsub(qy, m2_t);
            let qy_m2t_sq = builder.ins().fmul(qy_m2t, qy_m2t);
            let b_val = builder.ins().fma(m2, qx_ht_sq, qy_m2t_sq);

            // d2 = (qy.min(-qx*m2 - qy*0.5) > 0) ? 0 : min(a, b)  — branchless
            let neg_qx_m2 = builder.ins().fmul(neg_qx, m2);
            let half_qy = builder.ins().fmul(half, qy);
            let cond_val = builder.ins().fsub(neg_qx_m2, half_qy);
            let min_cond = builder.ins().fmin(qy, cond_val);
            let cmp_pos = builder.ins().fcmp(FloatCC::GreaterThan, min_cond, zero);
            let ab_min = builder.ins().fmin(a_val, b_val);
            let d2 = builder.ins().select(cmp_pos, zero, ab_min);

            // result = sqrt((d2 + qz²) * inv_m2) * sign(max(qz, -py))  — Division Exorcism
            let qz_sq = builder.ins().fmul(qz, qz);
            let d2_qz = builder.ins().fadd(d2, qz_sq);
            let d2_scaled = builder.ins().fmul(d2_qz, inv_m2);
            let dist = builder.ins().sqrt(d2_scaled);

            let neg_py = builder.ins().fneg(py);
            let sign_arg = builder.ins().fmax(qz, neg_py);
            let cmp_sign = builder.ins().fcmp(FloatCC::LessThan, sign_arg, zero);
            let sign = builder.ins().select(cmp_sign, neg_one, one);

            Ok(builder.ins().fmul(sign, dist))
        }

        SdfNode::Octahedron { size } => {
            let s = emitter.emit(builder, *size);
            let zero = builder.ins().f32const(0.0);
            let half = builder.ins().f32const(0.5);
            let three = builder.ins().f32const(3.0);
            let inv_sqrt3 = builder.ins().f32const(0.57735027);

            let abs_px = builder.ins().fabs(x);
            let abs_py = builder.ins().fabs(y);
            let abs_pz = builder.ins().fabs(z);

            // m = |px| + |py| + |pz| - s
            let sum_xy = builder.ins().fadd(abs_px, abs_py);
            let sum_xyz = builder.ins().fadd(sum_xy, abs_pz);
            let m = builder.ins().fsub(sum_xyz, s);

            // 4-way branchless: cascading selects
            let three_px = builder.ins().fmul(three, abs_px);
            let three_py = builder.ins().fmul(three, abs_py);
            let three_pz = builder.ins().fmul(three, abs_pz);

            let case1 = builder.ins().fcmp(FloatCC::LessThan, three_px, m);
            let case2 = builder.ins().fcmp(FloatCC::LessThan, three_py, m);
            let case3 = builder.ins().fcmp(FloatCC::LessThan, three_pz, m);

            // q for case3: (pz, px, py), for case2: (py, pz, px), for case1: (px, py, pz)
            let qx_c2 = builder.ins().select(case2, abs_py, abs_pz);
            let qy_c2 = builder.ins().select(case2, abs_pz, abs_px);
            let qz_c2 = builder.ins().select(case2, abs_px, abs_py);
            let qx = builder.ins().select(case1, abs_px, qx_c2);
            let qy = builder.ins().select(case1, abs_py, qy_c2);
            let qz = builder.ins().select(case1, abs_pz, qz_c2);

            // k = clamp(0.5 * (qz - qy + s), 0, s)
            let qz_qy = builder.ins().fsub(qz, qy);
            let qz_qy_s = builder.ins().fadd(qz_qy, s);
            let half_v = builder.ins().fmul(half, qz_qy_s);
            let k_min = builder.ins().fmin(half_v, s);
            let k = builder.ins().fmax(zero, k_min);

            // detail = length(qx, qy - s + k, qz - k)
            let qy_s = builder.ins().fsub(qy, s);
            let qy_s_k = builder.ins().fadd(qy_s, k);
            let qz_k = builder.ins().fsub(qz, k);
            let detail = emit_length3(builder, qx, qy_s_k, qz_k);

            // early = m * (1/sqrt(3))
            let early = builder.ins().fmul(m, inv_sqrt3);

            // result = any_case ? detail : early
            let any_12 = builder.ins().bor(case1, case2);
            let any_case = builder.ins().bor(any_12, case3);
            Ok(builder.ins().select(any_case, detail, early))
        }

        SdfNode::HexPrism { hex_radius, half_height } => {
            let hr = emitter.emit(builder, *hex_radius);
            let hh = emitter.emit(builder, *half_height);

            // Hex constants: k = (-sqrt(3)/2, 0.5, 1/sqrt(3))
            let kx = builder.ins().f32const(-0.8660254);
            let ky = builder.ins().f32const(0.5);
            let kz = builder.ins().f32const(0.57735027);
            let two = builder.ins().f32const(2.0);
            let zero = builder.ins().f32const(0.0);
            let neg_one = builder.ins().f32const(-1.0);
            let one = builder.ins().f32const(1.0);

            let abs_px = builder.ins().fabs(x);
            let abs_py = builder.ins().fabs(y);
            let abs_pz = builder.ins().fabs(z);

            // Hex symmetry reflection
            let ky_py = builder.ins().fmul(ky, abs_py);
            let dot_kxy = builder.ins().fma(kx, abs_px, ky_py);
            let dot_min = builder.ins().fmin(dot_kxy, zero);
            let reflect = builder.ins().fmul(two, dot_min);

            let reflect_kx = builder.ins().fmul(reflect, kx);
            let reflect_ky = builder.ins().fmul(reflect, ky);
            let px_r = builder.ins().fsub(abs_px, reflect_kx);
            let py_r = builder.ins().fsub(abs_py, reflect_ky);

            // clamped_x = clamp(px_r, -kz*hr, kz*hr)
            let kz_hr = builder.ins().fmul(kz, hr);
            let neg_kz_hr = builder.ins().fneg(kz_hr);
            let px_min = builder.ins().fmin(px_r, kz_hr);
            let clamped_x = builder.ins().fmax(neg_kz_hr, px_min);

            // d_xy = length(px_r - clamped_x, py_r - hr) * sign(py_r - hr)
            let dx = builder.ins().fsub(px_r, clamped_x);
            let dy = builder.ins().fsub(py_r, hr);
            let len_dxy = emit_length2(builder, dx, dy);
            let cmp_dy_neg = builder.ins().fcmp(FloatCC::LessThan, dy, zero);
            let sign_dy = builder.ins().select(cmp_dy_neg, neg_one, one);
            let d_xy = builder.ins().fmul(len_dxy, sign_dy);

            // d_z = |pz| - half_height
            let d_z = builder.ins().fsub(abs_pz, hh);

            // result = max(d_xy, d_z).min(0) + length(max(d_xy, 0), max(d_z, 0))
            let max_dxy_dz = builder.ins().fmax(d_xy, d_z);
            let interior = builder.ins().fmin(max_dxy_dz, zero);
            let d_xy_pos = builder.ins().fmax(d_xy, zero);
            let d_z_pos = builder.ins().fmax(d_z, zero);
            let exterior = emit_length2(builder, d_xy_pos, d_z_pos);

            Ok(builder.ins().fadd(interior, exterior))
        }

        SdfNode::Link { half_length, r1, r2 } => {
            let hl = emitter.emit(builder, *half_length);
            let r1_val = emitter.emit(builder, *r1);
            let r2_val = emitter.emit(builder, *r2);
            let zero = builder.ins().f32const(0.0);

            let abs_y = builder.ins().fabs(y);
            let y_sub = builder.ins().fsub(abs_y, hl);
            let qy = builder.ins().fmax(y_sub, zero);

            let xy_len = emit_length2(builder, x, qy);
            let xy_sub = builder.ins().fsub(xy_len, r1_val);

            let d_len = emit_length2(builder, xy_sub, z);
            Ok(builder.ins().fsub(d_len, r2_val))
        }

        // ============ Operations ============

        SdfNode::Union { a, b } => {
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            Ok(builder.ins().fmin(d_a, d_b))
        }

        SdfNode::Intersection { a, b } => {
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            Ok(builder.ins().fmax(d_a, d_b))
        }

        SdfNode::Subtraction { a, b } => {
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            let neg_b = builder.ins().fneg(d_b);
            Ok(builder.ins().fmax(d_a, neg_b))
        }

        SdfNode::SmoothUnion { a, b, k } => {
            // Division Exorcism: emit both k and inv_k
            let k_val = emitter.emit(builder, *k);
            let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / *k };
            let inv_k_val = emitter.emit(builder, inv_k);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            Ok(emit_smooth_min(builder, d_a, d_b, k_val, inv_k_val))
        }

        SdfNode::SmoothIntersection { a, b, k } => {
            let k_val = emitter.emit(builder, *k);
            let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / *k };
            let inv_k_val = emitter.emit(builder, inv_k);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            Ok(emit_smooth_max(builder, d_a, d_b, k_val, inv_k_val))
        }

        SdfNode::SmoothSubtraction { a, b, k } => {
            let k_val = emitter.emit(builder, *k);
            let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / *k };
            let inv_k_val = emitter.emit(builder, inv_k);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            let neg_b = builder.ins().fneg(d_b);
            Ok(emit_smooth_max(builder, d_a, neg_b, k_val, inv_k_val))
        }

        SdfNode::ChamferUnion { a, b, r } => {
            let r_val = emitter.emit(builder, *r);
            let s_val = emitter.emit(builder, std::f32::consts::FRAC_1_SQRT_2);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            Ok(emit_chamfer_min(builder, d_a, d_b, r_val, s_val))
        }

        SdfNode::ChamferIntersection { a, b, r } => {
            let r_val = emitter.emit(builder, *r);
            let s_val = emitter.emit(builder, std::f32::consts::FRAC_1_SQRT_2);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            let neg_a = builder.ins().fneg(d_a);
            let neg_b = builder.ins().fneg(d_b);
            let cm = emit_chamfer_min(builder, neg_a, neg_b, r_val, s_val);
            Ok(builder.ins().fneg(cm))
        }

        SdfNode::ChamferSubtraction { a, b, r } => {
            let r_val = emitter.emit(builder, *r);
            let s_val = emitter.emit(builder, std::f32::consts::FRAC_1_SQRT_2);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            let neg_a = builder.ins().fneg(d_a);
            let cm = emit_chamfer_min(builder, neg_a, d_b, r_val, s_val);
            Ok(builder.ins().fneg(cm))
        }

        SdfNode::StairsUnion { a, b, r, n } => {
            let r_val = emitter.emit(builder, *r);
            let n_val = emitter.emit(builder, *n);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            Ok(emit_stairs_min(builder, d_a, d_b, r_val, n_val))
        }

        SdfNode::StairsIntersection { a, b, r, n } => {
            let r_val = emitter.emit(builder, *r);
            let n_val = emitter.emit(builder, *n);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            let neg_a = builder.ins().fneg(d_a);
            let neg_b = builder.ins().fneg(d_b);
            let sm = emit_stairs_min(builder, neg_a, neg_b, r_val, n_val);
            Ok(builder.ins().fneg(sm))
        }

        SdfNode::StairsSubtraction { a, b, r, n } => {
            let r_val = emitter.emit(builder, *r);
            let n_val = emitter.emit(builder, *n);
            let d_a = compile_node(builder, a, x, y, z, emitter)?;
            let d_b = compile_node(builder, b, x, y, z, emitter)?;
            let neg_a = builder.ins().fneg(d_a);
            let sm = emit_stairs_min(builder, neg_a, d_b, r_val, n_val);
            Ok(builder.ins().fneg(sm))
        }

        // ============ Transforms ============

        SdfNode::Translate { child, offset } => {
            let ox = emitter.emit(builder, offset.x);
            let oy = emitter.emit(builder, offset.y);
            let oz = emitter.emit(builder, offset.z);

            let nx = builder.ins().fsub(x, ox);
            let ny = builder.ins().fsub(y, oy);
            let nz = builder.ins().fsub(z, oz);

            compile_node(builder, child, nx, ny, nz, emitter)
        }

        SdfNode::Rotate { child, rotation } => {
            let inv_rot = rotation.inverse();
            let qx = emitter.emit(builder, inv_rot.x);
            let qy = emitter.emit(builder, inv_rot.y);
            let qz = emitter.emit(builder, inv_rot.z);
            let qw = emitter.emit(builder, inv_rot.w);

            let (rx, ry, rz) = emit_quat_rotate(builder, x, y, z, qx, qy, qz, qw);
            compile_node(builder, child, rx, ry, rz, emitter)
        }

        SdfNode::Scale { child, factor } => {
            // Division Exorcism: pre-compute 1/factor
            let f = emitter.emit(builder, *factor);
            let inv_f = emitter.emit(builder, 1.0 / *factor);

            let nx = builder.ins().fmul(x, inv_f);
            let ny = builder.ins().fmul(y, inv_f);
            let nz = builder.ins().fmul(z, inv_f);

            let d = compile_node(builder, child, nx, ny, nz, emitter)?;
            Ok(builder.ins().fmul(d, f))
        }

        SdfNode::ScaleNonUniform { child, factors } => {
            // Division Exorcism: pre-compute reciprocals
            let inv_fx = emitter.emit(builder, 1.0 / factors.x);
            let inv_fy = emitter.emit(builder, 1.0 / factors.y);
            let inv_fz = emitter.emit(builder, 1.0 / factors.z);
            let min_scale = emitter.emit(builder, factors.x.min(factors.y).min(factors.z));

            let nx = builder.ins().fmul(x, inv_fx);
            let ny = builder.ins().fmul(y, inv_fy);
            let nz = builder.ins().fmul(z, inv_fz);

            let d = compile_node(builder, child, nx, ny, nz, emitter)?;
            Ok(builder.ins().fmul(d, min_scale))
        }

        // ============ Modifiers ============

        SdfNode::Twist { child, strength } => {
            let k = emitter.emit(builder, *strength);
            let angle = builder.ins().fmul(k, y);

            let (cos_a, sin_a) = emit_sincos_approx(builder, angle);

            let cos_x = builder.ins().fmul(cos_a, x);
            let sin_z = builder.ins().fmul(sin_a, z);
            let sin_x = builder.ins().fmul(sin_a, x);
            let cos_z = builder.ins().fmul(cos_a, z);

            let nx = builder.ins().fsub(cos_x, sin_z);
            let nz = builder.ins().fadd(sin_x, cos_z);

            compile_node(builder, child, nx, y, nz, emitter)
        }

        SdfNode::Bend { child, curvature } => {
            let k = emitter.emit(builder, *curvature);
            let angle = builder.ins().fmul(k, x);

            let (cos_a, sin_a) = emit_sincos_approx(builder, angle);

            let cos_y = builder.ins().fmul(cos_a, y);
            let sin_x = builder.ins().fmul(sin_a, x);
            let ny = builder.ins().fsub(cos_y, sin_x);
            let sin_y = builder.ins().fmul(sin_a, y);
            let cos_x = builder.ins().fmul(cos_a, x);
            let nx = builder.ins().fadd(sin_y, cos_x);

            compile_node(builder, child, nx, ny, z, emitter)
        }

        SdfNode::Round { child, radius } => {
            let d = compile_node(builder, child, x, y, z, emitter)?;
            let r = emitter.emit(builder, *radius);
            Ok(builder.ins().fsub(d, r))
        }

        SdfNode::Onion { child, thickness } => {
            let d = compile_node(builder, child, x, y, z, emitter)?;
            let t = emitter.emit(builder, *thickness);
            let abs_d = builder.ins().fabs(d);
            Ok(builder.ins().fsub(abs_d, t))
        }

        SdfNode::Elongate { child, amount } => {
            let sx = emitter.emit(builder, amount.x);
            let sy = emitter.emit(builder, amount.y);
            let sz = emitter.emit(builder, amount.z);
            let nsx = builder.ins().fneg(sx);
            let nsy = builder.ins().fneg(sy);
            let nsz = builder.ins().fneg(sz);

            let cx1 = builder.ins().fmin(x, sx);
            let cx = builder.ins().fmax(nsx, cx1);
            let cy1 = builder.ins().fmin(y, sy);
            let cy = builder.ins().fmax(nsy, cy1);
            let cz1 = builder.ins().fmin(z, sz);
            let cz = builder.ins().fmax(nsz, cz1);

            let qx = builder.ins().fsub(x, cx);
            let qy = builder.ins().fsub(y, cy);
            let qz = builder.ins().fsub(z, cz);

            compile_node(builder, child, qx, qy, qz, emitter)
        }

        SdfNode::RepeatInfinite { child, spacing } => {
            // Division Exorcism: pre-compute reciprocals for mod
            let half_px = emitter.emit(builder, spacing.x * 0.5);
            let half_py = emitter.emit(builder, spacing.y * 0.5);
            let half_pz = emitter.emit(builder, spacing.z * 0.5);
            let px = emitter.emit(builder, spacing.x);
            let py = emitter.emit(builder, spacing.y);
            let pz = emitter.emit(builder, spacing.z);
            let inv_px = emitter.emit(builder, 1.0 / spacing.x);
            let inv_py = emitter.emit(builder, 1.0 / spacing.y);
            let inv_pz = emitter.emit(builder, 1.0 / spacing.z);

            let x_off = builder.ins().fadd(x, half_px);
            let y_off = builder.ins().fadd(y, half_py);
            let z_off = builder.ins().fadd(z, half_pz);

            let nx = emit_mod_fast(builder, x_off, px, inv_px);
            let ny = emit_mod_fast(builder, y_off, py, inv_py);
            let nz = emit_mod_fast(builder, z_off, pz, inv_pz);

            let qx = builder.ins().fsub(nx, half_px);
            let qy = builder.ins().fsub(ny, half_py);
            let qz = builder.ins().fsub(nz, half_pz);

            compile_node(builder, child, qx, qy, qz, emitter)
        }

        SdfNode::RepeatFinite { child, count, spacing } => {
            // Division Exorcism: pre-compute reciprocals
            let inv_px = emitter.emit(builder, 1.0 / spacing.x);
            let inv_py = emitter.emit(builder, 1.0 / spacing.y);
            let inv_pz = emitter.emit(builder, 1.0 / spacing.z);
            let px = emitter.emit(builder, spacing.x);
            let py = emitter.emit(builder, spacing.y);
            let pz = emitter.emit(builder, spacing.z);
            let cx = emitter.emit(builder, count[0] as f32);
            let cy = emitter.emit(builder, count[1] as f32);
            let cz = emitter.emit(builder, count[2] as f32);
            let ncx = builder.ins().fneg(cx);
            let ncy = builder.ins().fneg(cy);
            let ncz = builder.ins().fneg(cz);

            // round(p * inv_spacing)  — Division Exorcism
            let rx = builder.ins().fmul(x, inv_px);
            let ry = builder.ins().fmul(y, inv_py);
            let rz = builder.ins().fmul(z, inv_pz);
            let rx = builder.ins().nearest(rx);
            let ry = builder.ins().nearest(ry);
            let rz = builder.ins().nearest(rz);

            // clamp to count
            let rx1 = builder.ins().fmin(rx, cx);
            let rx = builder.ins().fmax(ncx, rx1);
            let ry1 = builder.ins().fmin(ry, cy);
            let ry = builder.ins().fmax(ncy, ry1);
            let rz1 = builder.ins().fmin(rz, cz);
            let rz = builder.ins().fmax(ncz, rz1);

            // q = p - spacing * clamped
            let sx = builder.ins().fmul(px, rx);
            let sy = builder.ins().fmul(py, ry);
            let sz = builder.ins().fmul(pz, rz);
            let qx = builder.ins().fsub(x, sx);
            let qy = builder.ins().fsub(y, sy);
            let qz = builder.ins().fsub(z, sz);

            compile_node(builder, child, qx, qy, qz, emitter)
        }

        SdfNode::Mirror { child, axes } => {
            // Mirror axes are structural (compile-time decision)
            let nx = if axes.x != 0.0 { builder.ins().fabs(x) } else { x };
            let ny = if axes.y != 0.0 { builder.ins().fabs(y) } else { y };
            let nz = if axes.z != 0.0 { builder.ins().fabs(z) } else { z };
            compile_node(builder, child, nx, ny, nz, emitter)
        }

        SdfNode::Revolution { child, offset } => {
            let off = emitter.emit(builder, *offset);
            let len_xz = emit_length2(builder, x, z);
            let q = builder.ins().fsub(len_xz, off);
            let zero = builder.ins().f32const(0.0);
            compile_node(builder, child, q, y, zero, emitter)
        }

        SdfNode::Extrude { child, half_height } => {
            let zero = builder.ins().f32const(0.0);
            let child_d = compile_node(builder, child, x, y, zero, emitter)?;

            let h = emitter.emit(builder, *half_height);
            let abs_z = builder.ins().fabs(z);
            let w_y = builder.ins().fsub(abs_z, h);

            let inner = builder.ins().fmax(child_d, w_y);
            let inside = builder.ins().fmin(inner, zero);

            let mx = builder.ins().fmax(child_d, zero);
            let my = builder.ins().fmax(w_y, zero);
            let outside = emit_length2(builder, mx, my);

            Ok(builder.ins().fadd(inside, outside))
        }

        SdfNode::Noise { .. } => {
            Err(JitError::UnsupportedNode("Noise".to_string()))
        }

        SdfNode::SweepBezier { .. } => {
            Err(JitError::UnsupportedNode("SweepBezier".to_string()))
        }

        SdfNode::WithMaterial { child, .. } => {
            compile_node(builder, child, x, y, z, emitter)
        }

        // ============ Implementable Operations ============

        SdfNode::XOR { a, b } => {
            let da = compile_node(builder, a, x, y, z, emitter)?;
            let db = compile_node(builder, b, x, y, z, emitter)?;
            let min_ab = builder.ins().fmin(da, db);
            let max_ab = builder.ins().fmax(da, db);
            let neg_max = builder.ins().fneg(max_ab);
            Ok(builder.ins().fmax(min_ab, neg_max))
        }

        SdfNode::Morph { a, b, t } => {
            let da = compile_node(builder, a, x, y, z, emitter)?;
            let db = compile_node(builder, b, x, y, z, emitter)?;
            let t_val = emitter.emit(builder, *t);
            let diff = builder.ins().fsub(db, da);
            let blend = builder.ins().fmul(diff, t_val);
            Ok(builder.ins().fadd(da, blend))
        }

        SdfNode::Pipe { a, b, r } => {
            let da = compile_node(builder, a, x, y, z, emitter)?;
            let db = compile_node(builder, b, x, y, z, emitter)?;
            let r_val = emitter.emit(builder, *r);
            let len = emit_length2(builder, da, db);
            Ok(builder.ins().fsub(len, r_val))
        }

        SdfNode::Engrave { a, b, r } => {
            let da = compile_node(builder, a, x, y, z, emitter)?;
            let db = compile_node(builder, b, x, y, z, emitter)?;
            let r_val = emitter.emit(builder, *r);
            let abs_db = builder.ins().fabs(db);
            let sum = builder.ins().fadd(da, r_val);
            let diff = builder.ins().fsub(sum, abs_db);
            let half = builder.ins().f32const(0.5);
            let half_val = builder.ins().fmul(diff, half);
            Ok(builder.ins().fmax(da, half_val))
        }

        SdfNode::Groove { a, b, ra, rb } => {
            let da = compile_node(builder, a, x, y, z, emitter)?;
            let db = compile_node(builder, b, x, y, z, emitter)?;
            let ra_val = emitter.emit(builder, *ra);
            let rb_val = emitter.emit(builder, *rb);
            let abs_db = builder.ins().fabs(db);
            let sum = builder.ins().fadd(da, ra_val);
            let diff = builder.ins().fsub(rb_val, abs_db);
            let inner = builder.ins().fmin(sum, diff);
            Ok(builder.ins().fmax(da, inner))
        }

        SdfNode::Tongue { a, b, ra, rb } => {
            let da = compile_node(builder, a, x, y, z, emitter)?;
            let db = compile_node(builder, b, x, y, z, emitter)?;
            let ra_val = emitter.emit(builder, *ra);
            let rb_val = emitter.emit(builder, *rb);
            let abs_db = builder.ins().fabs(db);
            let sub_ra = builder.ins().fsub(da, ra_val);
            let sub_rb = builder.ins().fsub(abs_db, rb_val);
            let inner = builder.ins().fmax(sub_ra, sub_rb);
            Ok(builder.ins().fmin(da, inner))
        }

        // ============ Implementable Modifier ============

        SdfNode::OctantMirror { child } => {
            let ax = builder.ins().fabs(x);
            let ay = builder.ins().fabs(y);
            let az = builder.ins().fabs(z);
            // Sort: x >= y >= z
            let max_xy = builder.ins().fmax(ax, ay);
            let min_xy = builder.ins().fmin(ax, ay);
            let sx = builder.ins().fmax(max_xy, az);
            let mid_cand = builder.ins().fmin(max_xy, az);
            let sy = builder.ins().fmax(min_xy, mid_cand);
            let sz = builder.ins().fmin(min_xy, mid_cand);
            compile_node(builder, child, sx, sy, sz, emitter)
        }

        // ============ Unsupported Complex Primitives ============

        SdfNode::Triangle { .. }
        | SdfNode::Bezier { .. }
        | SdfNode::RoundedBox { .. }
        | SdfNode::CappedCone { .. }
        | SdfNode::CappedTorus { .. }
        | SdfNode::RoundedCylinder { .. }
        | SdfNode::TriangularPrism { .. }
        | SdfNode::CutSphere { .. }
        | SdfNode::CutHollowSphere { .. }
        | SdfNode::DeathStar { .. }
        | SdfNode::SolidAngle { .. }
        | SdfNode::Rhombus { .. }
        | SdfNode::Horseshoe { .. }
        | SdfNode::Vesica { .. }
        | SdfNode::InfiniteCylinder { .. }
        | SdfNode::InfiniteCone { .. }
        | SdfNode::Gyroid { .. }
        | SdfNode::Heart { .. }
        | SdfNode::Tube { .. }
        | SdfNode::Barrel { .. }
        | SdfNode::Diamond { .. }
        | SdfNode::ChamferedCube { .. }
        | SdfNode::SchwarzP { .. }
        | SdfNode::Superellipsoid { .. }
        | SdfNode::RoundedX { .. }
        | SdfNode::Pie { .. }
        | SdfNode::Trapezoid { .. }
        | SdfNode::Parallelogram { .. }
        | SdfNode::Tunnel { .. }
        | SdfNode::UnevenCapsule { .. }
        | SdfNode::Egg { .. }
        | SdfNode::ArcShape { .. }
        | SdfNode::Moon { .. }
        | SdfNode::CrossShape { .. }
        | SdfNode::BlobbyCross { .. }
        | SdfNode::ParabolaSegment { .. }
        | SdfNode::RegularPolygon { .. }
        | SdfNode::StarPolygon { .. }
        | SdfNode::Stairs { .. }
        | SdfNode::Helix { .. }
        | SdfNode::Tetrahedron { .. }
        | SdfNode::Dodecahedron { .. }
        | SdfNode::Icosahedron { .. }
        | SdfNode::TruncatedOctahedron { .. }
        | SdfNode::TruncatedIcosahedron { .. }
        | SdfNode::BoxFrame { .. }
        | SdfNode::DiamondSurface { .. }
        | SdfNode::Neovius { .. }
        | SdfNode::Lidinoid { .. }
        | SdfNode::IWP { .. }
        | SdfNode::FRD { .. }
        | SdfNode::FischerKochS { .. }
        | SdfNode::PMY { .. }
        => {
            Err(JitError::UnsupportedNode("ComplexPrimitive".to_string()))
        }

        // ============ Unsupported Complex Operations/Modifiers ============

        SdfNode::ColumnsUnion { .. }
        | SdfNode::ColumnsIntersection { .. }
        | SdfNode::ColumnsSubtraction { .. }
        | SdfNode::Taper { .. }
        | SdfNode::Displacement { .. }
        | SdfNode::PolarRepeat { .. }
        => {
            Err(JitError::UnsupportedNode("ComplexModifier".to_string()))
        }
    }
}

// ============ Helper Functions (Deep Fried v2) ============

/// Emit 3D vector length with FMA
fn emit_length3(builder: &mut FunctionBuilder, x: Value, y: Value, z: Value) -> Value {
    let zz = builder.ins().fmul(z, z);
    let yy_zz = builder.ins().fma(y, y, zz);
    let len_sq = builder.ins().fma(x, x, yy_zz);
    builder.ins().sqrt(len_sq)
}

/// Emit 2D vector length with FMA
fn emit_length2(builder: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    let yy = builder.ins().fmul(y, y);
    let len_sq = builder.ins().fma(x, x, yy);
    builder.ins().sqrt(len_sq)
}

/// Emit 3D dot product with FMA
fn emit_dot3(
    builder: &mut FunctionBuilder,
    ax: Value, ay: Value, az: Value,
    bx: Value, by: Value, bz: Value,
) -> Value {
    let zz = builder.ins().fmul(az, bz);
    let yy_zz = builder.ins().fma(ay, by, zz);
    builder.ins().fma(ax, bx, yy_zz)
}

/// Emit smooth minimum with Division Exorcism (uses pre-computed inv_k)
fn emit_smooth_min(
    builder: &mut FunctionBuilder,
    a: Value, b: Value,
    k: Value, inv_k: Value,
) -> Value {
    // h = max(k - abs(a - b), 0) * inv_k  — Division Exorcism!
    // return min(a, b) - h² * k * 0.25
    let zero = builder.ins().f32const(0.0);
    let quarter = builder.ins().f32const(0.25);

    let diff = builder.ins().fsub(a, b);
    let abs_diff = builder.ins().fabs(diff);
    let k_minus = builder.ins().fsub(k, abs_diff);
    let h_num = builder.ins().fmax(k_minus, zero);
    let h = builder.ins().fmul(h_num, inv_k); // Division Exorcism!

    let min_ab = builder.ins().fmin(a, b);
    let h2 = builder.ins().fmul(h, h);
    let h2k = builder.ins().fmul(h2, k);
    let correction = builder.ins().fmul(h2k, quarter);

    builder.ins().fsub(min_ab, correction)
}

/// Emit smooth maximum with Division Exorcism
fn emit_smooth_max(
    builder: &mut FunctionBuilder,
    a: Value, b: Value,
    k: Value, inv_k: Value,
) -> Value {
    let neg_a = builder.ins().fneg(a);
    let neg_b = builder.ins().fneg(b);
    let result = emit_smooth_min(builder, neg_a, neg_b, k, inv_k);
    builder.ins().fneg(result)
}

/// Emit chamfer minimum: min(min(a,b), (a+b)*s - r)
fn emit_chamfer_min(
    builder: &mut FunctionBuilder,
    a: Value, b: Value,
    r: Value, s: Value,
) -> Value {
    let min_ab = builder.ins().fmin(a, b);
    let sum = builder.ins().fadd(a, b);
    let scaled = builder.ins().fmul(sum, s);
    let chamfer = builder.ins().fsub(scaled, r);
    builder.ins().fmin(min_ab, chamfer)
}

/// Emit stairs minimum (Mercury hg_sdf fOpUnionStairs)
///
/// Implements the full stepped blend formula with 45-degree rotation,
/// modular repetition, and edge computation.
fn emit_stairs_min(
    builder: &mut FunctionBuilder,
    a: Value, b: Value,
    r: Value, n: Value,
) -> Value {
    let half = builder.ins().f32const(0.5);
    let s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
    let s2 = builder.ins().f32const(std::f32::consts::SQRT_2);

    // rn = r / n
    let rn = builder.ins().fdiv(r, n);
    // off = (r - rn) * 0.5 * sqrt(2)
    let r_minus_rn = builder.ins().fsub(r, rn);
    let tmp1 = builder.ins().fmul(r_minus_rn, half);
    let off = builder.ins().fmul(tmp1, s2);
    // step = r * sqrt(2) / n
    let r_s2 = builder.ins().fmul(r, s2);
    let step = builder.ins().fdiv(r_s2, n);

    // pR45: rotate (b-a, a+b) by 1/sqrt(2)
    // px = (b - a) * s - off
    let b_minus_a = builder.ins().fsub(b, a);
    let px_rot = builder.ins().fmul(b_minus_a, s);
    let px = builder.ins().fsub(px_rot, off);
    // py = (a + b) * s - off
    let a_plus_b = builder.ins().fadd(a, b);
    let py_rot = builder.ins().fmul(a_plus_b, s);
    let py = builder.ins().fsub(py_rot, off);

    // pMod1: px2 = px + 0.5 * sqrt(2) * rn
    let half_s2_rn = builder.ins().fmul(half, s2);
    let half_s2_rn2 = builder.ins().fmul(half_s2_rn, rn);
    let px2 = builder.ins().fadd(px, half_s2_rn2);
    // t = px2 + step * 0.5
    let step_half = builder.ins().fmul(step, half);
    let t = builder.ins().fadd(px2, step_half);
    // px3 = t - step * floor(t / step) - step * 0.5
    let t_div_step = builder.ins().fdiv(t, step);
    let t_floor = builder.ins().floor(t_div_step);
    let t_mod = builder.ins().fmul(step, t_floor);
    let t_sub = builder.ins().fsub(t, t_mod);
    let px3 = builder.ins().fsub(t_sub, step_half);

    // d2 = min(min(a, b), py)
    let min_ab = builder.ins().fmin(a, b);
    let d2 = builder.ins().fmin(min_ab, py);

    // Rotate back: npx = (px3 + py) * s, npy = (py - px3) * s
    let px3_plus_py = builder.ins().fadd(px3, py);
    let npx = builder.ins().fmul(px3_plus_py, s);
    let py_minus_px3 = builder.ins().fsub(py, px3);
    let npy = builder.ins().fmul(py_minus_px3, s);

    // edge = 0.5 * rn
    let edge = builder.ins().fmul(half, rn);

    // result = min(d2, max(npx - edge, npy - edge))
    let npx_e = builder.ins().fsub(npx, edge);
    let npy_e = builder.ins().fsub(npy, edge);
    let vmax = builder.ins().fmax(npx_e, npy_e);
    builder.ins().fmin(d2, vmax)
}

/// Emit quaternion rotation from pre-loaded Values
fn emit_quat_rotate(
    builder: &mut FunctionBuilder,
    x: Value, y: Value, z: Value,
    qx: Value, qy: Value, qz: Value, qw: Value,
) -> (Value, Value, Value) {
    let two = builder.ins().f32const(2.0);

    // t = 2 * cross(q.xyz, v)
    let qy_z = builder.ins().fmul(qy, z);
    let qz_y = builder.ins().fmul(qz, y);
    let cx = builder.ins().fsub(qy_z, qz_y);

    let qz_x = builder.ins().fmul(qz, x);
    let qx_z = builder.ins().fmul(qx, z);
    let cy = builder.ins().fsub(qz_x, qx_z);

    let qx_y = builder.ins().fmul(qx, y);
    let qy_x = builder.ins().fmul(qy, x);
    let cz = builder.ins().fsub(qx_y, qy_x);

    let tx = builder.ins().fmul(two, cx);
    let ty = builder.ins().fmul(two, cy);
    let tz = builder.ins().fmul(two, cz);

    // result = v + qw * t + cross(q.xyz, t)
    let qw_tx = builder.ins().fmul(qw, tx);
    let qw_ty = builder.ins().fmul(qw, ty);
    let qw_tz = builder.ins().fmul(qw, tz);

    let qy_tz = builder.ins().fmul(qy, tz);
    let qz_ty = builder.ins().fmul(qz, ty);
    let c2x = builder.ins().fsub(qy_tz, qz_ty);

    let qz_tx = builder.ins().fmul(qz, tx);
    let qx_tz = builder.ins().fmul(qx, tz);
    let c2y = builder.ins().fsub(qz_tx, qx_tz);

    let qx_ty = builder.ins().fmul(qx, ty);
    let qy_tx = builder.ins().fmul(qy, tx);
    let c2z = builder.ins().fsub(qx_ty, qy_tx);

    let x_qwtx = builder.ins().fadd(x, qw_tx);
    let rx = builder.ins().fadd(x_qwtx, c2x);

    let y_qwty = builder.ins().fadd(y, qw_ty);
    let ry = builder.ins().fadd(y_qwty, c2y);

    let z_qwtz = builder.ins().fadd(z, qw_tz);
    let rz = builder.ins().fadd(z_qwtz, c2z);

    (rx, ry, rz)
}

/// Emit approximate sin/cos using Taylor series
fn emit_sincos_approx(
    builder: &mut FunctionBuilder,
    angle: Value,
) -> (Value, Value) {
    let one = builder.ins().f32const(1.0);
    let half = builder.ins().f32const(0.5);
    let sixth = builder.ins().f32const(1.0 / 6.0);
    let twentyfourth = builder.ins().f32const(1.0 / 24.0);
    let one_twenty = builder.ins().f32const(1.0 / 120.0);

    let x2 = builder.ins().fmul(angle, angle);
    let x3 = builder.ins().fmul(x2, angle);
    let x4 = builder.ins().fmul(x2, x2);
    let x5 = builder.ins().fmul(x4, angle);

    // sin(x) = x - x³/6 + x⁵/120
    let sin_term1 = builder.ins().fmul(x3, sixth);
    let sin_term2 = builder.ins().fmul(x5, one_twenty);
    let sin_result = builder.ins().fsub(angle, sin_term1);
    let sin_result = builder.ins().fadd(sin_result, sin_term2);

    // cos(x) = 1 - x²/2 + x⁴/24
    let cos_term1 = builder.ins().fmul(x2, half);
    let cos_term2 = builder.ins().fmul(x4, twentyfourth);
    let cos_result = builder.ins().fsub(one, cos_term1);
    let cos_result = builder.ins().fadd(cos_result, cos_term2);

    (cos_result, sin_result)
}

/// Emit modulo with Division Exorcism: a mod b = a - b * floor(a * inv_b)
fn emit_mod_fast(
    builder: &mut FunctionBuilder,
    a: Value, b: Value, inv_b: Value,
) -> Value {
    let div = builder.ins().fmul(a, inv_b); // Division Exorcism!
    let floored = builder.ins().floor(div);
    let mult = builder.ins().fmul(b, floored);
    builder.ins().fsub(a, mult)
}

// ============ Parameter Extraction ============

/// Extract parameters from an SDF tree in the same order as dynamic compilation.
///
/// This function must push params in EXACTLY the same order as `compile_node`
/// calls `emitter.emit()`. Used by `JitCompiledSdfDynamic::update_params()`.
pub fn extract_jit_params(node: &SdfNode) -> Vec<f32> {
    let mut params = Vec::new();
    extract_params_recursive(node, &mut params);
    params
}

fn extract_params_recursive(node: &SdfNode, params: &mut Vec<f32>) {
    match node {
        // Primitives
        SdfNode::Sphere { radius } => {
            params.push(*radius);
        }

        SdfNode::Box3d { half_extents } => {
            params.push(half_extents.x);
            params.push(half_extents.y);
            params.push(half_extents.z);
        }

        SdfNode::Cylinder { radius, half_height } => {
            params.push(*radius);
            params.push(*half_height);
        }

        SdfNode::Torus { major_radius, minor_radius } => {
            params.push(*major_radius);
            params.push(*minor_radius);
        }

        SdfNode::Plane { normal, distance } => {
            params.push(normal.x);
            params.push(normal.y);
            params.push(normal.z);
            params.push(*distance);
        }

        SdfNode::Capsule { point_a, point_b, radius } => {
            params.push(point_a.x);
            params.push(point_a.y);
            params.push(point_a.z);
            params.push(point_b.x);
            params.push(point_b.y);
            params.push(point_b.z);
            params.push(*radius);
            // inv_dot_ba_ba (Division Exorcism derived)
            let ba = *point_b - *point_a;
            let dbb = ba.dot(ba);
            params.push(if dbb.abs() < 1e-10 { 1.0 } else { 1.0 / dbb });
        }

        SdfNode::Ellipsoid { radii } => {
            params.push(1.0 / radii.x);
            params.push(1.0 / radii.y);
            params.push(1.0 / radii.z);
            params.push(1.0 / (radii.x * radii.x));
            params.push(1.0 / (radii.y * radii.y));
            params.push(1.0 / (radii.z * radii.z));
        }

        SdfNode::Link { half_length, r1, r2 } => {
            params.push(*half_length);
            params.push(*r1);
            params.push(*r2);
        }

        // Operations (no params for Union/Intersection/Subtraction)
        SdfNode::Union { a, b } | SdfNode::Intersection { a, b } | SdfNode::Subtraction { a, b } => {
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
        }

        SdfNode::SmoothUnion { a, b, k }
        | SdfNode::SmoothIntersection { a, b, k }
        | SdfNode::SmoothSubtraction { a, b, k } => {
            params.push(*k);
            params.push(if k.abs() < 1e-10 { 1.0 } else { 1.0 / *k });
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
        }

        SdfNode::ChamferUnion { a, b, r }
        | SdfNode::ChamferIntersection { a, b, r }
        | SdfNode::ChamferSubtraction { a, b, r } => {
            params.push(*r);
            params.push(std::f32::consts::FRAC_1_SQRT_2);
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
        }

        SdfNode::StairsUnion { a, b, r, n }
        | SdfNode::StairsIntersection { a, b, r, n }
        | SdfNode::StairsSubtraction { a, b, r, n } => {
            params.push(*r);
            params.push(*n);
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
        }

        // Transforms
        SdfNode::Translate { child, offset } => {
            params.push(offset.x);
            params.push(offset.y);
            params.push(offset.z);
            extract_params_recursive(child, params);
        }

        SdfNode::Rotate { child, rotation } => {
            let inv_rot = rotation.inverse();
            params.push(inv_rot.x);
            params.push(inv_rot.y);
            params.push(inv_rot.z);
            params.push(inv_rot.w);
            extract_params_recursive(child, params);
        }

        SdfNode::Scale { child, factor } => {
            params.push(*factor);
            params.push(1.0 / *factor);
            extract_params_recursive(child, params);
        }

        SdfNode::ScaleNonUniform { child, factors } => {
            params.push(1.0 / factors.x);
            params.push(1.0 / factors.y);
            params.push(1.0 / factors.z);
            params.push(factors.x.min(factors.y).min(factors.z));
            extract_params_recursive(child, params);
        }

        // Modifiers
        SdfNode::Twist { child, strength } => {
            params.push(*strength);
            extract_params_recursive(child, params);
        }

        SdfNode::Bend { child, curvature } => {
            params.push(*curvature);
            extract_params_recursive(child, params);
        }

        SdfNode::Round { child, radius } => {
            extract_params_recursive(child, params);
            params.push(*radius);
        }

        SdfNode::Onion { child, thickness } => {
            extract_params_recursive(child, params);
            params.push(*thickness);
        }

        SdfNode::Elongate { child, amount } => {
            params.push(amount.x);
            params.push(amount.y);
            params.push(amount.z);
            extract_params_recursive(child, params);
        }

        SdfNode::RepeatInfinite { child, spacing } => {
            params.push(spacing.x * 0.5);
            params.push(spacing.y * 0.5);
            params.push(spacing.z * 0.5);
            params.push(spacing.x);
            params.push(spacing.y);
            params.push(spacing.z);
            params.push(1.0 / spacing.x);
            params.push(1.0 / spacing.y);
            params.push(1.0 / spacing.z);
            extract_params_recursive(child, params);
        }

        SdfNode::RepeatFinite { child, count, spacing } => {
            params.push(1.0 / spacing.x);
            params.push(1.0 / spacing.y);
            params.push(1.0 / spacing.z);
            params.push(spacing.x);
            params.push(spacing.y);
            params.push(spacing.z);
            params.push(count[0] as f32);
            params.push(count[1] as f32);
            params.push(count[2] as f32);
            extract_params_recursive(child, params);
        }

        SdfNode::Mirror { child, .. } => {
            // axes are structural, not parametric
            extract_params_recursive(child, params);
        }

        SdfNode::Revolution { child, offset } => {
            params.push(*offset);
            extract_params_recursive(child, params);
        }

        SdfNode::SweepBezier { child, p0, p1, p2 } => {
            params.push(p0.x); params.push(p0.y);
            params.push(p1.x); params.push(p1.y);
            params.push(p2.x); params.push(p2.y);
            extract_params_recursive(child, params);
        }

        SdfNode::Extrude { child, half_height } => {
            extract_params_recursive(child, params);
            params.push(*half_height);
        }

        SdfNode::Cone { radius, half_height } => {
            params.push(*radius);
            params.push(*half_height);
            // Division Exorcism: pre-compute inv_k2_dot
            let k2d = radius * radius + (2.0 * half_height) * (2.0 * half_height);
            params.push(if k2d.abs() < 1e-10 { 1.0 } else { 1.0 / k2d });
        }

        SdfNode::RoundedCone { r1, r2, half_height } => {
            params.push(*r1);
            params.push(*r2);
            params.push(*half_height);
            let h_val = half_height * 2.0;
            let inv_h = if h_val.abs() < 1e-10 { 1.0 } else { 1.0 / h_val };
            let b_val = (r1 - r2) * inv_h;
            let a_val = (1.0 - b_val * b_val).max(0.0).sqrt();
            let ah_val = a_val * h_val;
            params.push(h_val);
            params.push(b_val);
            params.push(a_val);
            params.push(ah_val);
        }

        SdfNode::Pyramid { half_height } => {
            params.push(*half_height);
            let h_val = half_height * 2.0;
            let m2_val = h_val * h_val + 0.25;
            params.push(h_val);
            params.push(m2_val);
            params.push(1.0 / m2_val);
            params.push(1.0 / (m2_val + 0.25));
        }

        SdfNode::Octahedron { size } => {
            params.push(*size);
        }

        SdfNode::HexPrism { hex_radius, half_height } => {
            params.push(*hex_radius);
            params.push(*half_height);
        }

        // No-op / error nodes
        SdfNode::Noise { .. } => {}

        SdfNode::WithMaterial { child, .. } => {
            extract_params_recursive(child, params);
        }

        // Implementable operations
        SdfNode::XOR { a, b } => {
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
        }

        SdfNode::Morph { a, b, t } => {
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
            params.push(*t);
        }

        SdfNode::Pipe { a, b, r } => {
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
            params.push(*r);
        }

        SdfNode::Engrave { a, b, r } => {
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
            params.push(*r);
        }

        SdfNode::Groove { a, b, ra, rb } => {
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
            params.push(*ra);
            params.push(*rb);
        }

        SdfNode::Tongue { a, b, ra, rb } => {
            extract_params_recursive(a, params);
            extract_params_recursive(b, params);
            params.push(*ra);
            params.push(*rb);
        }

        // Implementable modifier
        SdfNode::OctantMirror { child } => {
            extract_params_recursive(child, params);
        }

        // Unsupported complex primitives (no params to extract)
        SdfNode::Triangle { .. }
        | SdfNode::Bezier { .. }
        | SdfNode::RoundedBox { .. }
        | SdfNode::CappedCone { .. }
        | SdfNode::CappedTorus { .. }
        | SdfNode::RoundedCylinder { .. }
        | SdfNode::TriangularPrism { .. }
        | SdfNode::CutSphere { .. }
        | SdfNode::CutHollowSphere { .. }
        | SdfNode::DeathStar { .. }
        | SdfNode::SolidAngle { .. }
        | SdfNode::Rhombus { .. }
        | SdfNode::Horseshoe { .. }
        | SdfNode::Vesica { .. }
        | SdfNode::InfiniteCylinder { .. }
        | SdfNode::InfiniteCone { .. }
        | SdfNode::Gyroid { .. }
        | SdfNode::Heart { .. }
        | SdfNode::Tube { .. }
        | SdfNode::Barrel { .. }
        | SdfNode::Diamond { .. }
        | SdfNode::ChamferedCube { .. }
        | SdfNode::SchwarzP { .. }
        | SdfNode::Superellipsoid { .. }
        | SdfNode::RoundedX { .. }
        | SdfNode::Pie { .. }
        | SdfNode::Trapezoid { .. }
        | SdfNode::Parallelogram { .. }
        | SdfNode::Tunnel { .. }
        | SdfNode::UnevenCapsule { .. }
        | SdfNode::Egg { .. }
        | SdfNode::ArcShape { .. }
        | SdfNode::Moon { .. }
        | SdfNode::CrossShape { .. }
        | SdfNode::BlobbyCross { .. }
        | SdfNode::ParabolaSegment { .. }
        | SdfNode::RegularPolygon { .. }
        | SdfNode::StarPolygon { .. }
        | SdfNode::Stairs { .. }
        | SdfNode::Helix { .. }
        | SdfNode::Tetrahedron { .. }
        | SdfNode::Dodecahedron { .. }
        | SdfNode::Icosahedron { .. }
        | SdfNode::TruncatedOctahedron { .. }
        | SdfNode::TruncatedIcosahedron { .. }
        | SdfNode::BoxFrame { .. }
        | SdfNode::DiamondSurface { .. }
        | SdfNode::Neovius { .. }
        | SdfNode::Lidinoid { .. }
        | SdfNode::IWP { .. }
        | SdfNode::FRD { .. }
        | SdfNode::FischerKochS { .. }
        | SdfNode::PMY { .. }
        | SdfNode::ColumnsUnion { .. }
        | SdfNode::ColumnsIntersection { .. }
        | SdfNode::ColumnsSubtraction { .. }
        | SdfNode::Taper { .. }
        | SdfNode::Displacement { .. }
        | SdfNode::PolarRepeat { .. }
        => {}
    }
}
