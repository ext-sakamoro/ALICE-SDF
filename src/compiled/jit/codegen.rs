//! JIT Code Generation: SDF to Cranelift IR
//!
//! This module converts SDF node trees to Cranelift Intermediate Representation (IR),
//! which is then compiled to native machine code.
//!
//! Author: Moroya Sakamoto

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, UserFuncName};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};

use crate::types::SdfNode;
use super::runtime::JitError;

/// JIT Compiler for SDF evaluation
///
/// Compiles SDF node trees to native machine code using Cranelift.
pub struct JitCompiler<'a> {
    module: &'a mut JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl<'a> JitCompiler<'a> {
    /// Create a new JIT compiler
    pub fn new(module: &'a mut JITModule) -> Self {
        let ctx = module.make_context();
        let func_ctx = FunctionBuilderContext::new();

        JitCompiler {
            module,
            ctx,
            func_ctx,
        }
    }

    /// Compile an SDF node to a function
    pub fn compile_sdf(&mut self, node: &SdfNode) -> Result<FuncId, JitError> {
        // Create function signature: fn(f32, f32, f32) -> f32
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32)); // x
        sig.params.push(AbiParam::new(types::F32)); // y
        sig.params.push(AbiParam::new(types::F32)); // z
        sig.returns.push(AbiParam::new(types::F32)); // distance

        // Declare the function
        let func_id = self
            .module
            .declare_function("sdf_eval", Linkage::Export, &sig)
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        // Set up the function
        self.ctx.func.signature = sig;
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // Build the function body
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);

            // Create entry block
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Get parameters
            let x = builder.block_params(entry_block)[0];
            let y = builder.block_params(entry_block)[1];
            let z = builder.block_params(entry_block)[2];

            // Generate code for the SDF
            let result = compile_node(&mut builder, node, x, y, z)?;

            // Return the result
            builder.ins().return_(&[result]);

            builder.finalize();
        }

        // Define the function
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JitError::CompilationError(e.to_string()))?;

        // Clear the context for reuse
        self.module.clear_context(&mut self.ctx);

        Ok(func_id)
    }
}

/// Compile a single SDF node, returning the IR value for the distance
fn compile_node(
    builder: &mut FunctionBuilder,
    node: &SdfNode,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
    z: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, JitError> {
    match node {
        // ============ Primitives ============

        SdfNode::Sphere { radius } => {
            // distance = length(p) - radius
            let len = emit_length3(builder, x, y, z);
            let r = builder.ins().f32const(*radius);
            Ok(builder.ins().fsub(len, r))
        }

        SdfNode::Box3d { half_extents } => {
            // q = abs(p) - half_extents
            // distance = length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0)
            let hx = builder.ins().f32const(half_extents.x);
            let hy = builder.ins().f32const(half_extents.y);
            let hz = builder.ins().f32const(half_extents.z);

            let ax = builder.ins().fabs(x);
            let ay = builder.ins().fabs(y);
            let az = builder.ins().fabs(z);

            let qx = builder.ins().fsub(ax, hx);
            let qy = builder.ins().fsub(ay, hy);
            let qz = builder.ins().fsub(az, hz);

            let zero = builder.ins().f32const(0.0);

            // max(q, 0)
            let mx = builder.ins().fmax(qx, zero);
            let my = builder.ins().fmax(qy, zero);
            let mz = builder.ins().fmax(qz, zero);

            // length(max(q, 0))
            let outside = emit_length3(builder, mx, my, mz);

            // min(max(q.x, max(q.y, q.z)), 0)
            let inner_max = builder.ins().fmax(qy, qz);
            let outer_max = builder.ins().fmax(qx, inner_max);
            let inside = builder.ins().fmin(outer_max, zero);

            Ok(builder.ins().fadd(outside, inside))
        }

        SdfNode::Cylinder { radius, half_height } => {
            // d.x = length(p.xz) - radius
            // d.y = abs(p.y) - half_height
            // return min(max(d.x, d.y), 0) + length(max(d, 0))
            let r = builder.ins().f32const(*radius);
            let h = builder.ins().f32const(*half_height);
            let zero = builder.ins().f32const(0.0);

            // length(p.xz)
            let len_xz = emit_length2(builder, x, z);
            let dx = builder.ins().fsub(len_xz, r);

            let ay = builder.ins().fabs(y);
            let dy = builder.ins().fsub(ay, h);

            // min(max(d.x, d.y), 0)
            let inner = builder.ins().fmax(dx, dy);
            let inside = builder.ins().fmin(inner, zero);

            // length(max(d, 0))
            let mx = builder.ins().fmax(dx, zero);
            let my = builder.ins().fmax(dy, zero);
            let outside = emit_length2(builder, mx, my);

            Ok(builder.ins().fadd(inside, outside))
        }

        SdfNode::Torus {
            major_radius,
            minor_radius,
        } => {
            // q = vec2(length(p.xz) - major_radius, p.y)
            // return length(q) - minor_radius
            let major = builder.ins().f32const(*major_radius);
            let minor = builder.ins().f32const(*minor_radius);

            let len_xz = emit_length2(builder, x, z);
            let qx = builder.ins().fsub(len_xz, major);

            let len_q = emit_length2(builder, qx, y);
            Ok(builder.ins().fsub(len_q, minor))
        }

        SdfNode::Plane { normal, distance } => {
            // dot(p, normal) + distance
            let nx = builder.ins().f32const(normal.x);
            let ny = builder.ins().f32const(normal.y);
            let nz = builder.ins().f32const(normal.z);
            let dist = builder.ins().f32const(*distance);

            let dot = emit_dot3(builder, x, y, z, nx, ny, nz);
            Ok(builder.ins().fadd(dot, dist))
        }

        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => {
            // pa = p - a, ba = b - a
            // h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
            // return length(pa - ba * h) - radius
            let ax = builder.ins().f32const(point_a.x);
            let ay = builder.ins().f32const(point_a.y);
            let az = builder.ins().f32const(point_a.z);
            let bx = builder.ins().f32const(point_b.x);
            let by = builder.ins().f32const(point_b.y);
            let bz = builder.ins().f32const(point_b.z);
            let r = builder.ins().f32const(*radius);

            // pa = p - a
            let pax = builder.ins().fsub(x, ax);
            let pay = builder.ins().fsub(y, ay);
            let paz = builder.ins().fsub(z, az);

            // ba = b - a
            let bax = builder.ins().fsub(bx, ax);
            let bay = builder.ins().fsub(by, ay);
            let baz = builder.ins().fsub(bz, az);

            // h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
            let dot_pa_ba = emit_dot3(builder, pax, pay, paz, bax, bay, baz);
            let dot_ba_ba = emit_dot3(builder, bax, bay, baz, bax, bay, baz);
            let h_raw = builder.ins().fdiv(dot_pa_ba, dot_ba_ba);
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

        // ============ Operations ============

        SdfNode::Union { a, b } => {
            let d_a = compile_node(builder, a, x, y, z)?;
            let d_b = compile_node(builder, b, x, y, z)?;
            Ok(builder.ins().fmin(d_a, d_b))
        }

        SdfNode::Intersection { a, b } => {
            let d_a = compile_node(builder, a, x, y, z)?;
            let d_b = compile_node(builder, b, x, y, z)?;
            Ok(builder.ins().fmax(d_a, d_b))
        }

        SdfNode::Subtraction { a, b } => {
            // max(a, -b)
            let d_a = compile_node(builder, a, x, y, z)?;
            let d_b = compile_node(builder, b, x, y, z)?;
            let neg_b = builder.ins().fneg(d_b);
            Ok(builder.ins().fmax(d_a, neg_b))
        }

        SdfNode::SmoothUnion { a, b, k } => {
            let k_val = builder.ins().f32const(*k);
            let d_a = compile_node(builder, a, x, y, z)?;
            let d_b = compile_node(builder, b, x, y, z)?;
            Ok(emit_smooth_min(builder, d_a, d_b, k_val))
        }

        SdfNode::SmoothIntersection { a, b, k } => {
            let k_val = builder.ins().f32const(*k);
            let d_a = compile_node(builder, a, x, y, z)?;
            let d_b = compile_node(builder, b, x, y, z)?;
            Ok(emit_smooth_max(builder, d_a, d_b, k_val))
        }

        SdfNode::SmoothSubtraction { a, b, k } => {
            let k_val = builder.ins().f32const(*k);
            let d_a = compile_node(builder, a, x, y, z)?;
            let d_b = compile_node(builder, b, x, y, z)?;
            let neg_b = builder.ins().fneg(d_b);
            Ok(emit_smooth_max(builder, d_a, neg_b, k_val))
        }

        // ============ Transforms ============

        SdfNode::Translate { child, offset } => {
            // p' = p - offset
            let ox = builder.ins().f32const(offset.x);
            let oy = builder.ins().f32const(offset.y);
            let oz = builder.ins().f32const(offset.z);

            let nx = builder.ins().fsub(x, ox);
            let ny = builder.ins().fsub(y, oy);
            let nz = builder.ins().fsub(z, oz);

            compile_node(builder, child, nx, ny, nz)
        }

        SdfNode::Rotate { child, rotation } => {
            // Apply inverse rotation to point
            let inv_rot = rotation.inverse();

            // Quaternion rotation: p' = q^-1 * p * q
            let (rx, ry, rz) = emit_quat_rotate(builder, x, y, z, inv_rot);
            compile_node(builder, child, rx, ry, rz)
        }

        SdfNode::Scale { child, factor } => {
            // p' = p / factor, distance *= factor
            let f = builder.ins().f32const(*factor);
            let inv_f = builder.ins().f32const(1.0 / *factor);

            let nx = builder.ins().fmul(x, inv_f);
            let ny = builder.ins().fmul(y, inv_f);
            let nz = builder.ins().fmul(z, inv_f);

            let d = compile_node(builder, child, nx, ny, nz)?;
            Ok(builder.ins().fmul(d, f))
        }

        SdfNode::ScaleNonUniform { child, factors } => {
            // p' = p / factors
            let inv_fx = builder.ins().f32const(1.0 / factors.x);
            let inv_fy = builder.ins().f32const(1.0 / factors.y);
            let inv_fz = builder.ins().f32const(1.0 / factors.z);

            let nx = builder.ins().fmul(x, inv_fx);
            let ny = builder.ins().fmul(y, inv_fy);
            let nz = builder.ins().fmul(z, inv_fz);

            let d = compile_node(builder, child, nx, ny, nz)?;

            // Approximate distance correction using minimum scale
            let min_scale = factors.x.min(factors.y).min(factors.z);
            let scale = builder.ins().f32const(min_scale);
            Ok(builder.ins().fmul(d, scale))
        }

        // ============ Modifiers ============

        SdfNode::Twist { child, strength } => {
            // angle = strength * y
            // cos_a, sin_a = cos(angle), sin(angle)
            // x' = cos_a * x - sin_a * z
            // z' = sin_a * x + cos_a * z
            let k = builder.ins().f32const(*strength);
            let angle = builder.ins().fmul(k, y);

            // Approximate sin/cos using Taylor series
            let (cos_a, sin_a) = emit_sincos_approx(builder, angle);

            let cos_x = builder.ins().fmul(cos_a, x);
            let sin_z = builder.ins().fmul(sin_a, z);
            let sin_x = builder.ins().fmul(sin_a, x);
            let cos_z = builder.ins().fmul(cos_a, z);

            let nx = builder.ins().fsub(cos_x, sin_z);
            let nz = builder.ins().fadd(sin_x, cos_z);

            compile_node(builder, child, nx, y, nz)
        }

        SdfNode::Bend { child, curvature } => {
            // Similar to twist but bends around x axis
            let k = builder.ins().f32const(*curvature);
            let angle = builder.ins().fmul(k, x);

            let (cos_a, sin_a) = emit_sincos_approx(builder, angle);

            let cos_y = builder.ins().fmul(cos_a, y);
            let sin_x = builder.ins().fmul(sin_a, x);

            let ny = builder.ins().fsub(cos_y, sin_x);
            let sin_y = builder.ins().fmul(sin_a, y);
            let cos_x = builder.ins().fmul(cos_a, x);
            let nx = builder.ins().fadd(sin_y, cos_x);

            compile_node(builder, child, nx, ny, z)
        }

        SdfNode::Round { child, radius } => {
            let d = compile_node(builder, child, x, y, z)?;
            let r = builder.ins().f32const(*radius);
            Ok(builder.ins().fsub(d, r))
        }

        SdfNode::Onion { child, thickness } => {
            let d = compile_node(builder, child, x, y, z)?;
            let t = builder.ins().f32const(*thickness);
            let abs_d = builder.ins().fabs(d);
            Ok(builder.ins().fsub(abs_d, t))
        }

        SdfNode::Elongate { child, amount } => {
            // q = p - clamp(p, -amount, amount)
            let sx = builder.ins().f32const(amount.x);
            let sy = builder.ins().f32const(amount.y);
            let sz = builder.ins().f32const(amount.z);
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

            compile_node(builder, child, qx, qy, qz)
        }

        SdfNode::RepeatInfinite { child, spacing } => {
            // q = mod(p + 0.5 * spacing, spacing) - 0.5 * spacing
            let half_px = builder.ins().f32const(spacing.x * 0.5);
            let half_py = builder.ins().f32const(spacing.y * 0.5);
            let half_pz = builder.ins().f32const(spacing.z * 0.5);
            let px = builder.ins().f32const(spacing.x);
            let py = builder.ins().f32const(spacing.y);
            let pz = builder.ins().f32const(spacing.z);

            let x_off = builder.ins().fadd(x, half_px);
            let y_off = builder.ins().fadd(y, half_py);
            let z_off = builder.ins().fadd(z, half_pz);

            let nx = emit_mod(builder, x_off, px);
            let ny = emit_mod(builder, y_off, py);
            let nz = emit_mod(builder, z_off, pz);

            let qx = builder.ins().fsub(nx, half_px);
            let qy = builder.ins().fsub(ny, half_py);
            let qz = builder.ins().fsub(nz, half_pz);

            compile_node(builder, child, qx, qy, qz)
        }

        SdfNode::RepeatFinite {
            child,
            count,
            spacing,
        } => {
            // q = p - spacing * clamp(round(p / spacing), -count, count)
            let px = builder.ins().f32const(spacing.x);
            let py = builder.ins().f32const(spacing.y);
            let pz = builder.ins().f32const(spacing.z);
            let cx = builder.ins().f32const(count[0] as f32);
            let cy = builder.ins().f32const(count[1] as f32);
            let cz = builder.ins().f32const(count[2] as f32);
            let ncx = builder.ins().fneg(cx);
            let ncy = builder.ins().fneg(cy);
            let ncz = builder.ins().fneg(cz);

            // round(p / spacing)
            let rx = builder.ins().fdiv(x, px);
            let ry = builder.ins().fdiv(y, py);
            let rz = builder.ins().fdiv(z, pz);
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

            compile_node(builder, child, qx, qy, qz)
        }

        SdfNode::Noise { .. } => {
            // Noise is complex to JIT compile efficiently
            Err(JitError::UnsupportedNode("Noise".to_string()))
        }
    }
}

// ============ Helper Functions ============

/// Emit code for 3D vector length
fn emit_length3(
    builder: &mut FunctionBuilder,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
    z: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let xx = builder.ins().fmul(x, x);
    let yy = builder.ins().fmul(y, y);
    let zz = builder.ins().fmul(z, z);
    let sum_xy = builder.ins().fadd(xx, yy);
    let sum = builder.ins().fadd(sum_xy, zz);
    builder.ins().sqrt(sum)
}

/// Emit code for 2D vector length
fn emit_length2(
    builder: &mut FunctionBuilder,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let xx = builder.ins().fmul(x, x);
    let yy = builder.ins().fmul(y, y);
    let sum = builder.ins().fadd(xx, yy);
    builder.ins().sqrt(sum)
}

/// Emit code for 3D dot product
fn emit_dot3(
    builder: &mut FunctionBuilder,
    ax: cranelift_codegen::ir::Value,
    ay: cranelift_codegen::ir::Value,
    az: cranelift_codegen::ir::Value,
    bx: cranelift_codegen::ir::Value,
    by: cranelift_codegen::ir::Value,
    bz: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let xx = builder.ins().fmul(ax, bx);
    let yy = builder.ins().fmul(ay, by);
    let zz = builder.ins().fmul(az, bz);
    let sum_xy = builder.ins().fadd(xx, yy);
    builder.ins().fadd(sum_xy, zz)
}

/// Emit smooth minimum (polynomial)
fn emit_smooth_min(
    builder: &mut FunctionBuilder,
    a: cranelift_codegen::ir::Value,
    b: cranelift_codegen::ir::Value,
    k: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    // h = max(k - abs(a - b), 0) / k
    // return min(a, b) - h * h * k * 0.25
    let zero = builder.ins().f32const(0.0);
    let quarter = builder.ins().f32const(0.25);

    let diff = builder.ins().fsub(a, b);
    let abs_diff = builder.ins().fabs(diff);
    let k_minus = builder.ins().fsub(k, abs_diff);
    let h_num = builder.ins().fmax(k_minus, zero);
    let h = builder.ins().fdiv(h_num, k);

    let min_ab = builder.ins().fmin(a, b);
    let h2 = builder.ins().fmul(h, h);
    let h2k = builder.ins().fmul(h2, k);
    let correction = builder.ins().fmul(h2k, quarter);

    builder.ins().fsub(min_ab, correction)
}

/// Emit smooth maximum (polynomial)
fn emit_smooth_max(
    builder: &mut FunctionBuilder,
    a: cranelift_codegen::ir::Value,
    b: cranelift_codegen::ir::Value,
    k: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    // -smooth_min(-a, -b, k)
    let neg_a = builder.ins().fneg(a);
    let neg_b = builder.ins().fneg(b);
    let result = emit_smooth_min(builder, neg_a, neg_b, k);
    builder.ins().fneg(result)
}

/// Emit quaternion rotation
fn emit_quat_rotate(
    builder: &mut FunctionBuilder,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
    z: cranelift_codegen::ir::Value,
    quat: glam::Quat,
) -> (
    cranelift_codegen::ir::Value,
    cranelift_codegen::ir::Value,
    cranelift_codegen::ir::Value,
) {
    // Pre-compute quaternion constants
    let qx = builder.ins().f32const(quat.x);
    let qy = builder.ins().f32const(quat.y);
    let qz = builder.ins().f32const(quat.z);
    let qw = builder.ins().f32const(quat.w);

    // t = 2 * cross(q.xyz, v)
    // cross(q.xyz, v) = (qy*vz - qz*vy, qz*vx - qx*vz, qx*vy - qy*vx)
    let two = builder.ins().f32const(2.0);

    // cross x: qy*z - qz*y
    let qy_z = builder.ins().fmul(qy, z);
    let qz_y = builder.ins().fmul(qz, y);
    let cx = builder.ins().fsub(qy_z, qz_y);

    // cross y: qz*x - qx*z
    let qz_x = builder.ins().fmul(qz, x);
    let qx_z = builder.ins().fmul(qx, z);
    let cy = builder.ins().fsub(qz_x, qx_z);

    // cross z: qx*y - qy*x
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

    // cross(q.xyz, t)
    // cross x: qy*tz - qz*ty
    let qy_tz = builder.ins().fmul(qy, tz);
    let qz_ty = builder.ins().fmul(qz, ty);
    let c2x = builder.ins().fsub(qy_tz, qz_ty);

    // cross y: qz*tx - qx*tz
    let qz_tx = builder.ins().fmul(qz, tx);
    let qx_tz = builder.ins().fmul(qx, tz);
    let c2y = builder.ins().fsub(qz_tx, qx_tz);

    // cross z: qx*ty - qy*tx
    let qx_ty = builder.ins().fmul(qx, ty);
    let qy_tx = builder.ins().fmul(qy, tx);
    let c2z = builder.ins().fsub(qx_ty, qy_tx);

    // rx = x + qw*tx + c2x
    let x_qwtx = builder.ins().fadd(x, qw_tx);
    let rx = builder.ins().fadd(x_qwtx, c2x);

    // ry = y + qw*ty + c2y
    let y_qwty = builder.ins().fadd(y, qw_ty);
    let ry = builder.ins().fadd(y_qwty, c2y);

    // rz = z + qw*tz + c2z
    let z_qwtz = builder.ins().fadd(z, qw_tz);
    let rz = builder.ins().fadd(z_qwtz, c2z);

    (rx, ry, rz)
}

/// Emit approximate sin/cos using Taylor series
fn emit_sincos_approx(
    builder: &mut FunctionBuilder,
    angle: cranelift_codegen::ir::Value,
) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
    // Use 5th order Taylor approximation
    // sin(x) ≈ x - x³/6 + x⁵/120
    // cos(x) ≈ 1 - x²/2 + x⁴/24

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

/// Emit modulo operation (a mod b)
fn emit_mod(
    builder: &mut FunctionBuilder,
    a: cranelift_codegen::ir::Value,
    b: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    // a - b * floor(a / b)
    let div = builder.ins().fdiv(a, b);
    let floored = builder.ins().floor(div);
    let mult = builder.ins().fmul(b, floored);
    builder.ins().fsub(a, mult)
}
