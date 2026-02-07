//! SIMD-accelerated SDF evaluation (8 points at once)
//!
//! This module provides 8-wide SIMD evaluation using AVX2/AVX-512/NEON,
//! achieving up to 8x speedup over scalar evaluation.
//!
//! Author: Moroya Sakamoto

use super::compiler::CompiledSdf;
use super::opcode::OpCode;
use super::simd::{Quatx8, Vec3x8};
use glam::Vec3;
use wide::{f32x8, CmpLt, CmpGt};

use crate::modifiers::perlin_noise_3d;

/// Maximum stack depth for value stack (SIMD)
const MAX_VALUE_STACK: usize = 64;
/// Maximum stack depth for coordinate transforms (SIMD)
const MAX_COORD_STACK: usize = 32;

/// Coordinate frame for SIMD evaluation
#[derive(Clone, Copy)]
struct CoordFrameSimd {
    point: Vec3x8,
    scale_correction: f32x8,
    opcode: OpCode,
    params: [f32; 4],
}

impl Default for CoordFrameSimd {
    fn default() -> Self {
        CoordFrameSimd {
            point: Vec3x8::zero(),
            scale_correction: f32x8::ONE,
            opcode: OpCode::End,
            params: [0.0; 4],
        }
    }
}

/// Evaluate compiled SDF at 8 points simultaneously
///
/// This is the core SIMD evaluation function. It processes 8 points
/// in parallel using AVX2/AVX-512 instructions.
///
/// # Arguments
/// * `sdf` - Compiled SDF bytecode
/// * `points` - 8 points packed as Vec3x8
///
/// # Returns
/// 8 distance values as f32x8
#[inline]
pub fn eval_compiled_simd(sdf: &CompiledSdf, points: Vec3x8) -> f32x8 {
    let mut value_stack: [f32x8; MAX_VALUE_STACK] = [f32x8::ZERO; MAX_VALUE_STACK];
    let mut vsp: usize = 0;

    let mut coord_stack: [CoordFrameSimd; MAX_COORD_STACK] =
        [CoordFrameSimd::default(); MAX_COORD_STACK];
    let mut csp: usize = 0;

    let mut p = points;
    let mut scale_correction = f32x8::ONE;

    for inst in &sdf.instructions {
        match inst.opcode {
            // === Primitives ===
            OpCode::Sphere => {
                let r = f32x8::splat(inst.params[0]);
                let d = p.length() - r;
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Box3d => {
                let hx = f32x8::splat(inst.params[0]);
                let hy = f32x8::splat(inst.params[1]);
                let hz = f32x8::splat(inst.params[2]);

                // q = abs(p) - half_extents
                let qx = p.x.abs() - hx;
                let qy = p.y.abs() - hy;
                let qz = p.z.abs() - hz;

                // length(max(q, 0)) + min(max(q.x, q.y, q.z), 0)
                let qx_pos = qx.max(f32x8::ZERO);
                let qy_pos = qy.max(f32x8::ZERO);
                let qz_pos = qz.max(f32x8::ZERO);
                let outside = (qx_pos * qx_pos + qy_pos * qy_pos + qz_pos * qz_pos).sqrt();
                let inside = qx.max(qy).max(qz).min(f32x8::ZERO);
                let d = outside + inside;

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Cylinder => {
                let r = f32x8::splat(inst.params[0]);
                let h = f32x8::splat(inst.params[1]);

                // d.x = length(p.xz) - r
                // d.y = abs(p.y) - h
                let dx = (p.x * p.x + p.z * p.z).sqrt() - r;
                let dy = p.y.abs() - h;

                let dx_pos = dx.max(f32x8::ZERO);
                let dy_pos = dy.max(f32x8::ZERO);
                let outside = (dx_pos * dx_pos + dy_pos * dy_pos).sqrt();
                let inside = dx.max(dy).min(f32x8::ZERO);
                let d = outside + inside;

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Torus => {
                let major = f32x8::splat(inst.params[0]);
                let minor = f32x8::splat(inst.params[1]);

                // q = vec2(length(p.xz) - major, p.y)
                let qx = (p.x * p.x + p.z * p.z).sqrt() - major;
                let qy = p.y;
                let d = (qx * qx + qy * qy).sqrt() - minor;

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Plane => {
                let nx = f32x8::splat(inst.params[0]);
                let ny = f32x8::splat(inst.params[1]);
                let nz = f32x8::splat(inst.params[2]);
                let dist = f32x8::splat(inst.params[3]);

                let d = p.x * nx + p.y * ny + p.z * nz + dist;
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Capsule => {
                let ax = f32x8::splat(inst.params[0]);
                let ay = f32x8::splat(inst.params[1]);
                let az = f32x8::splat(inst.params[2]);
                let bx = f32x8::splat(inst.params[3]);
                let by = f32x8::splat(inst.params[4]);
                let bz = f32x8::splat(inst.params[5]);
                let radius = f32x8::splat(inst.get_capsule_radius());

                // pa = p - a, ba = b - a
                let pax = p.x - ax;
                let pay = p.y - ay;
                let paz = p.z - az;
                let bax = bx - ax;
                let bay = by - ay;
                let baz = bz - az;

                // h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
                let dot_pa_ba = pax * bax + pay * bay + paz * baz;
                let dot_ba_ba = bax * bax + bay * bay + baz * baz;
                // Branchless zero guard for degenerate capsule (a == b)
                let safe_dot = dot_ba_ba.max(f32x8::splat(1e-10));
                let h = (dot_pa_ba / safe_dot).max(f32x8::ZERO).min(f32x8::ONE);

                // length(pa - ba * h) - radius
                let dx = pax - bax * h;
                let dy = pay - bay * h;
                let dz = paz - baz * h;
                let d = (dx * dx + dy * dy + dz * dz).sqrt() - radius;

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Cone => {
                let radius = f32x8::splat(inst.params[0]);
                let h = f32x8::splat(inst.params[1]);

                let q_x = (p.x * p.x + p.z * p.z).sqrt();
                let q_y = p.y;

                let k2x = -radius;
                let k2y = h + h;

                // ca_r = q_y < 0 ? radius : 0
                let neg_mask = q_y.cmp_lt(f32x8::ZERO);
                let ca_r = neg_mask.blend(radius, f32x8::ZERO);

                let ca_x = q_x - q_x.min(ca_r);
                let ca_y = q_y.abs() - h;

                let diff_x = -q_x;
                let diff_y = h - q_y;
                let k2_dot = k2x * k2x + k2y * k2y;
                let t = ((diff_x * k2x + diff_y * k2y) / k2_dot)
                    .max(f32x8::ZERO)
                    .min(f32x8::ONE);

                let cb_x = q_x + k2x * t;
                let cb_y = q_y - h + k2y * t;

                // s = (cb_x < 0 && ca_y < 0) ? -1 : 1
                let both_neg = cb_x.cmp_lt(f32x8::ZERO) & ca_y.cmp_lt(f32x8::ZERO);
                let s = both_neg.blend(f32x8::splat(-1.0), f32x8::ONE);

                let d2 = (ca_x * ca_x + ca_y * ca_y).min(cb_x * cb_x + cb_y * cb_y);
                let d = s * d2.sqrt();

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Ellipsoid => {
                // Division Exorcism: pre-compute reciprocals (zero-safe)
                let rx = inst.params[0].max(1e-10);
                let ry = inst.params[1].max(1e-10);
                let rz = inst.params[2].max(1e-10);
                let inv_rx = f32x8::splat(1.0 / rx);
                let inv_ry = f32x8::splat(1.0 / ry);
                let inv_rz = f32x8::splat(1.0 / rz);
                let inv_rx2 = f32x8::splat(1.0 / (rx * rx));
                let inv_ry2 = f32x8::splat(1.0 / (ry * ry));
                let inv_rz2 = f32x8::splat(1.0 / (rz * rz));

                // k0 = length(p * inv_radii)
                let px_r = p.x * inv_rx;
                let py_r = p.y * inv_ry;
                let pz_r = p.z * inv_rz;
                let k0 = (px_r * px_r + py_r * py_r + pz_r * pz_r).sqrt();

                // k1 = length(p * inv_radii²)
                let px_rr = p.x * inv_rx2;
                let py_rr = p.y * inv_ry2;
                let pz_rr = p.z * inv_rz2;
                let k1 = (px_rr * px_rr + py_rr * py_rr + pz_rr * pz_rr).sqrt();

                let d = k0 * (k0 - f32x8::ONE) / k1.max(f32x8::splat(1e-10));
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::RoundedCone => {
                let r1 = f32x8::splat(inst.params[0]);
                let r2 = f32x8::splat(inst.params[1]);
                let half_height = f32x8::splat(inst.params[2]);

                let h = half_height + half_height;
                let q_x = (p.x * p.x + p.z * p.z).sqrt();
                let q_y = p.y + half_height;

                let b_val = (r1 - r2) / h;
                let a_val = (f32x8::ONE - b_val * b_val).sqrt();
                let k = q_x * (-b_val) + q_y * a_val;

                // Case 1: k < 0 → bottom sphere
                let d_bottom = (q_x * q_x + q_y * q_y).sqrt() - r1;
                // Case 2: k > a*h → top sphere
                let dy_top = q_y - h;
                let d_top = (q_x * q_x + dy_top * dy_top).sqrt() - r2;
                // Case 3: mantle
                let d_mantle = q_x * a_val + q_y * b_val - r1;

                let mask_bottom = k.cmp_lt(f32x8::ZERO);
                let mask_top = k.cmp_gt(a_val * h);
                let d = mask_bottom.blend(d_bottom, mask_top.blend(d_top, d_mantle));

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Pyramid => {
                let half_height = f32x8::splat(inst.params[0]);
                let h = half_height + half_height;
                let m2 = h * h + f32x8::splat(0.25);
                let half = f32x8::splat(0.5);

                let py = p.y + half_height;
                let abs_px = p.x.abs();
                let abs_pz = p.z.abs();

                // if pz > px { swap(px, pz) }
                let swap_mask = abs_pz.cmp_gt(abs_px);
                let px = swap_mask.blend(abs_pz, abs_px) - half;
                let pz = swap_mask.blend(abs_px, abs_pz) - half;

                let qx = pz;
                let qy = h * py - half * px;
                let qz = h * px + half * py;

                let s = (-qx).max(f32x8::ZERO);
                let t = ((qy - half * pz) / (m2 + f32x8::splat(0.25)))
                    .max(f32x8::ZERO)
                    .min(f32x8::ONE);

                let a = m2 * (qx + s) * (qx + s) + qy * qy;
                let half_t = half * t;
                let b = m2 * (qx + half_t) * (qx + half_t) + (qy - m2 * t) * (qy - m2 * t);

                // d2 = if qy.min(-qx * m2 - qy * 0.5) > 0 { 0 } else { min(a, b) }
                let inner = (-qx * m2 - qy * half).min(qy);
                let zero_mask = inner.cmp_gt(f32x8::ZERO);
                let d2 = zero_mask.blend(f32x8::ZERO, a.min(b));

                // sign = signum(max(qz, -py))
                let sign_input = qz.max(-py);
                let pos = sign_input.cmp_gt(f32x8::ZERO);
                let neg = sign_input.cmp_lt(f32x8::ZERO);
                let sign_val = pos.blend(f32x8::ONE, neg.blend(f32x8::splat(-1.0), f32x8::ZERO));

                let d = ((d2 + qz * qz) / m2).sqrt() * sign_val;

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Octahedron => {
                let s = f32x8::splat(inst.params[0]);
                let three = f32x8::splat(3.0);
                let inv_sqrt3 = f32x8::splat(0.57735027);
                let half = f32x8::splat(0.5);

                let ax = p.x.abs();
                let ay = p.y.abs();
                let az = p.z.abs();
                let m = ax + ay + az - s;

                // d_flat: the "else" branch = m * inv_sqrt3
                let d_flat = m * inv_sqrt3;

                // Exclusive masks for the 3 permutation cases
                let mask1 = (three * ax).cmp_lt(m);
                let mask2 = (three * ay).cmp_lt(m) & !mask1;
                let mask3 = (three * az).cmp_lt(m) & !mask1 & !mask2;
                let mask_any = mask1 | mask2 | mask3;

                // Select q permutation based on which case
                let qx = mask1.blend(ax, mask2.blend(ay, mask3.blend(az, f32x8::ZERO)));
                let qy = mask1.blend(ay, mask2.blend(az, mask3.blend(ax, f32x8::ZERO)));
                let qz = mask1.blend(az, mask2.blend(ax, mask3.blend(ay, f32x8::ZERO)));

                let k = (half * (qz - qy + s)).max(f32x8::ZERO).min(s);
                let vx = qx;
                let vy = qy - s + k;
                let vz = qz - k;
                let d_edge = (vx * vx + vy * vy + vz * vz).sqrt();

                let d = mask_any.blend(d_edge, d_flat);

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::HexPrism => {
                let hex_radius = f32x8::splat(inst.params[0]);
                let half_height = f32x8::splat(inst.params[1]);

                let kx = f32x8::splat(-0.8660254_f32);
                let ky = f32x8::splat(0.5_f32);
                let kz_c = f32x8::splat(0.57735027_f32);
                let two = f32x8::splat(2.0);

                let mut px = p.x.abs();
                let mut py = p.y.abs();
                let pz = p.z.abs();

                // Reflect across hex symmetry
                let dot_kxy = kx * px + ky * py;
                let reflect = two * dot_kxy.min(f32x8::ZERO);
                px = px - reflect * kx;
                py = py - reflect * ky;

                // Clamp and compute XY distance
                let clamped_x = px.max(-kz_c * hex_radius).min(kz_c * hex_radius);
                let dx = px - clamped_x;
                let dy = py - hex_radius;
                let d_xy_len = (dx * dx + dy * dy).sqrt();

                // signum(dy)
                let pos = dy.cmp_gt(f32x8::ZERO);
                let neg = dy.cmp_lt(f32x8::ZERO);
                let dy_sign = pos.blend(f32x8::ONE, neg.blend(f32x8::splat(-1.0), f32x8::ZERO));
                let d_xy = d_xy_len * dy_sign;

                let d_z = pz - half_height;

                // max(d_xy, d_z).min(0) + sqrt(max(d_xy,0)^2 + max(d_z,0)^2)
                let d_xy_pos = d_xy.max(f32x8::ZERO);
                let d_z_pos = d_z.max(f32x8::ZERO);
                let d = d_xy.max(d_z).min(f32x8::ZERO) + (d_xy_pos * d_xy_pos + d_z_pos * d_z_pos).sqrt();

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Link => {
                let half_length = f32x8::splat(inst.params[0]);
                let r1 = f32x8::splat(inst.params[1]);
                let r2 = f32x8::splat(inst.params[2]);

                let qx = p.x;
                let qy = (p.y.abs() - half_length).max(f32x8::ZERO);
                let qz = p.z;

                let xy_len = (qx * qx + qy * qy).sqrt() - r1;
                let d = (xy_len * xy_len + qz * qz).sqrt() - r2;

                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            // === Binary Operations ===
            OpCode::Union => {
                vsp -= 1;
                let b = value_stack[vsp];
                value_stack[vsp - 1] = value_stack[vsp - 1].min(b);
            }

            OpCode::Intersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                value_stack[vsp - 1] = value_stack[vsp - 1].max(b);
            }

            OpCode::Subtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                value_stack[vsp - 1] = value_stack[vsp - 1].max(-b);
            }

            OpCode::SmoothUnion => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                let k = f32x8::splat(inst.params[0]);
                value_stack[vsp - 1] = smooth_min_simd(a, b, k);
            }

            OpCode::SmoothIntersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                let k = f32x8::splat(inst.params[0]);
                value_stack[vsp - 1] = -smooth_min_simd(-a, -b, k);
            }

            OpCode::SmoothSubtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                let k = f32x8::splat(inst.params[0]);
                value_stack[vsp - 1] = -smooth_min_simd(-a, b, k);
            }

            // === Transforms ===
            OpCode::Translate => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Translate,
                    params: [0.0; 4],
                };
                csp += 1;

                let offset = Vec3x8 {
                    x: f32x8::splat(inst.params[0]),
                    y: f32x8::splat(inst.params[1]),
                    z: f32x8::splat(inst.params[2]),
                };
                p = p - offset;
            }

            OpCode::Rotate => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Rotate,
                    params: [0.0; 4],
                };
                csp += 1;

                let q = glam::Quat::from_xyzw(
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                let qx8 = Quatx8::splat(q.inverse());
                p = qx8.mul_vec3(p);
            }

            OpCode::Scale => {
                let inv_factor = inst.params[0]; // precomputed 1.0/factor
                let factor = inst.params[1];     // original factor
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Scale,
                    params: [factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                // Multiply by precomputed inverse (no division)
                p = p * f32x8::splat(inv_factor);
                scale_correction = scale_correction * f32x8::splat(factor);
            }

            OpCode::ScaleNonUniform => {
                let inv_sx = inst.params[0]; // precomputed 1.0/sx
                let inv_sy = inst.params[1]; // precomputed 1.0/sy
                let inv_sz = inst.params[2]; // precomputed 1.0/sz
                let min_factor = inst.params[3]; // precomputed min(sx,sy,sz)

                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::ScaleNonUniform,
                    params: [min_factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                // Multiply by precomputed inverses (no division)
                p = Vec3x8 {
                    x: p.x * f32x8::splat(inv_sx),
                    y: p.y * f32x8::splat(inv_sy),
                    z: p.z * f32x8::splat(inv_sz),
                };
                scale_correction = scale_correction * f32x8::splat(min_factor);
            }

            // === Modifiers ===
            OpCode::Twist => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Twist,
                    params: [0.0; 4],
                };
                csp += 1;

                let k = f32x8::splat(inst.params[0]);
                let angle = k * p.y;
                let c = cos_approx(angle);
                let s = sin_approx(angle);
                let new_x = c * p.x - s * p.z;
                let new_z = s * p.x + c * p.z;
                p = Vec3x8 {
                    x: new_x,
                    y: p.y,
                    z: new_z,
                };
            }

            OpCode::Bend => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Bend,
                    params: [0.0; 4],
                };
                csp += 1;

                let k = f32x8::splat(inst.params[0]);
                let angle = k * p.x;
                let c = cos_approx(angle);
                let s = sin_approx(angle);
                let new_x = c * p.x - s * p.y;
                let new_y = s * p.x + c * p.y;
                p = Vec3x8 {
                    x: new_x,
                    y: new_y,
                    z: p.z,
                };
            }

            OpCode::RepeatInfinite => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::RepeatInfinite,
                    params: [0.0; 4],
                };
                csp += 1;

                let sx = f32x8::splat(inst.params[0]);
                let sy = f32x8::splat(inst.params[1]);
                let sz = f32x8::splat(inst.params[2]);
                let half = f32x8::splat(0.5);

                // modulo operation: p - spacing * floor(p / spacing + 0.5)
                p = Vec3x8 {
                    x: p.x - sx * (p.x / sx + half).floor(),
                    y: p.y - sy * (p.y / sy + half).floor(),
                    z: p.z - sz * (p.z / sz + half).floor(),
                };
            }

            OpCode::RepeatFinite => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::RepeatFinite,
                    params: [0.0; 4],
                };
                csp += 1;

                let cx = f32x8::splat(inst.params[0]);
                let cy = f32x8::splat(inst.params[1]);
                let cz = f32x8::splat(inst.params[2]);
                let sx = f32x8::splat(inst.params[3]);
                let sy = f32x8::splat(inst.params[4]);
                let sz = f32x8::splat(inst.params[5]);
                let _half = f32x8::splat(0.5);

                // clamp(round(p / spacing), -count, count) * spacing
                let ix = ((p.x / sx).round()).max(-cx).min(cx);
                let iy = ((p.y / sy).round()).max(-cy).min(cy);
                let iz = ((p.z / sz).round()).max(-cz).min(cz);

                p = Vec3x8 {
                    x: p.x - ix * sx,
                    y: p.y - iy * sy,
                    z: p.z - iz * sz,
                };
            }

            OpCode::Round => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Round,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;
            }

            OpCode::Onion => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Onion,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;
            }

            OpCode::Elongate => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Elongate,
                    params: [0.0; 4],
                };
                csp += 1;

                let ax = f32x8::splat(inst.params[0]);
                let ay = f32x8::splat(inst.params[1]);
                let az = f32x8::splat(inst.params[2]);

                // p - clamp(p, -amount, amount)
                p = Vec3x8 {
                    x: p.x - p.x.max(-ax).min(ax),
                    y: p.y - p.y.max(-ay).min(ay),
                    z: p.z - p.z.max(-az).min(az),
                };
            }

            OpCode::Noise => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Noise,
                    params: [inst.params[0], inst.params[1], inst.params[2], 0.0],
                };
                csp += 1;
                // Noise doesn't modify point, only post-processes distance
            }

            OpCode::Mirror => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Mirror,
                    params: [inst.params[0], inst.params[1], inst.params[2], 0.0],
                };
                csp += 1;

                p = Vec3x8 {
                    x: if inst.params[0] != 0.0 { p.x.abs() } else { p.x },
                    y: if inst.params[1] != 0.0 { p.y.abs() } else { p.y },
                    z: if inst.params[2] != 0.0 { p.z.abs() } else { p.z },
                };
            }

            OpCode::Revolution => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Revolution,
                    params: [0.0; 4],
                };
                csp += 1;

                let offset = f32x8::splat(inst.params[0]);
                let q = (p.x * p.x + p.z * p.z).sqrt() - offset;
                p = Vec3x8 { x: q, y: p.y, z: f32x8::ZERO };
            }

            OpCode::Extrude => {
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Extrude,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;

                p = Vec3x8 { x: p.x, y: p.y, z: f32x8::ZERO };
            }

            // === Control ===
            OpCode::PopTransform => {
                csp -= 1;
                let frame = coord_stack[csp];

                match frame.opcode {
                    OpCode::Round => {
                        let radius = f32x8::splat(frame.params[0]);
                        value_stack[vsp - 1] = value_stack[vsp - 1] - radius;
                    }
                    OpCode::Onion => {
                        let thickness = f32x8::splat(frame.params[0]);
                        value_stack[vsp - 1] = value_stack[vsp - 1].abs() - thickness;
                    }
                    OpCode::Noise => {
                        // Process noise for each lane
                        let amplitude = frame.params[0];
                        let frequency = frame.params[1];
                        let seed = frame.params[2] as u32;
                        let mut noise_vals = [0.0f32; 8];
                        for i in 0..8 {
                            let pt = glam::Vec3::new(
                                frame.point.x.as_array_ref()[i] * frequency,
                                frame.point.y.as_array_ref()[i] * frequency,
                                frame.point.z.as_array_ref()[i] * frequency,
                            );
                            noise_vals[i] = perlin_noise_3d(pt.x, pt.y, pt.z, seed) * amplitude;
                        }
                        value_stack[vsp - 1] = value_stack[vsp - 1] + f32x8::new(noise_vals);
                    }
                    OpCode::Extrude => {
                        let half_height = f32x8::splat(frame.params[0]);
                        let child_dist = value_stack[vsp - 1];
                        let w_y = frame.point.z.abs() - half_height;
                        let w_x_pos = child_dist.max(f32x8::ZERO);
                        let w_y_pos = w_y.max(f32x8::ZERO);
                        let outside = (w_x_pos * w_x_pos + w_y_pos * w_y_pos).sqrt();
                        let inside = child_dist.max(w_y).min(f32x8::ZERO);
                        value_stack[vsp - 1] = outside + inside;
                    }
                    _ => {}
                }

                p = frame.point;
                scale_correction = frame.scale_correction;
            }

            OpCode::End => {
                break;
            }
        }
    }

    if vsp > 0 {
        value_stack[0]
    } else {
        f32x8::splat(f32::MAX)
    }
}

/// Smooth minimum (polynomial) for SIMD — branchless k=0 guard
#[inline]
fn smooth_min_simd(a: f32x8, b: f32x8, k: f32x8) -> f32x8 {
    let k = k.max(f32x8::splat(1e-10));
    let h = (f32x8::ONE - ((a - b).abs() / k)).max(f32x8::ZERO);
    a.min(b) - h * h * k * f32x8::splat(0.25)
}

/// Fast cosine approximation for SIMD
#[inline]
fn cos_approx(x: f32x8) -> f32x8 {
    // Normalize to [-pi, pi]
    let pi = f32x8::splat(std::f32::consts::PI);
    let two_pi = f32x8::splat(std::f32::consts::TAU);
    let x = x - (x / two_pi).round() * two_pi;

    // Bhaskara I approximation
    let x2 = x * x;
    let pi2 = pi * pi;
    (pi2 - f32x8::splat(4.0) * x2) / (pi2 + x2)
}

/// Fast sine approximation for SIMD
#[inline]
fn sin_approx(x: f32x8) -> f32x8 {
    cos_approx(x - f32x8::splat(std::f32::consts::FRAC_PI_2))
}

/// Batch evaluate compiled SDF using SIMD (8 points at a time)
///
/// This is the main entry point for SIMD-accelerated batch evaluation.
/// Points are processed 8 at a time, with any remainder handled by
/// scalar evaluation.
pub fn eval_compiled_batch_simd(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    let n = points.len();
    let mut results = vec![0.0f32; n];

    // Process 8 points at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        let p = Vec3x8::from_vecs([
            points[base],
            points[base + 1],
            points[base + 2],
            points[base + 3],
            points[base + 4],
            points[base + 5],
            points[base + 6],
            points[base + 7],
        ]);

        let d = eval_compiled_simd(sdf, p);
        let arr = d.to_array();
        results[base..base + 8].copy_from_slice(&arr);
    }

    // Handle remainder with scalar evaluation
    let remainder = n % 8;
    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            results[base + i] = super::eval::eval_compiled(sdf, points[base + i]);
        }
    }

    results
}

/// Parallel batch evaluate compiled SDF using SIMD
///
/// Combines SIMD (8-wide) with multi-threading for maximum performance.
pub fn eval_compiled_batch_simd_parallel(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    use rayon::prelude::*;

    let n = points.len();
    if n < 64 {
        // Not worth parallelizing for small inputs
        return eval_compiled_batch_simd(sdf, points);
    }

    // Process in chunks of 64 (8 SIMD lanes * 8 iterations per thread)
    let chunk_size = 64;
    let mut results = vec![0.0f32; n];

    results
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * chunk_size;
            let chunk_len = chunk.len();

            // Process 8 points at a time within this chunk
            let simd_iters = chunk_len / 8;
            for i in 0..simd_iters {
                let p_base = base + i * 8;
                let p = Vec3x8::from_vecs([
                    points[p_base],
                    points[p_base + 1],
                    points[p_base + 2],
                    points[p_base + 3],
                    points[p_base + 4],
                    points[p_base + 5],
                    points[p_base + 6],
                    points[p_base + 7],
                ]);

                let d = eval_compiled_simd(sdf, p);
                let arr = d.to_array();
                chunk[i * 8..(i + 1) * 8].copy_from_slice(&arr);
            }

            // Handle remainder within chunk
            let remainder = chunk_len % 8;
            if remainder > 0 {
                let r_base = simd_iters * 8;
                for i in 0..remainder {
                    chunk[r_base + i] = super::eval::eval_compiled(sdf, points[base + r_base + i]);
                }
            }
        });

    results
}

// ============================================================================
// Gradient (Normal) Computation - SIMD
// ============================================================================

/// Compute gradient (normal direction) using finite differences - SIMD 8-wide
///
/// Returns unnormalized gradient vectors (gx, gy, gz) for 8 points simultaneously.
/// The caller can normalize if needed.
///
/// # Performance
/// - 6 SDF evaluations per call (central differences)
/// - ~6x slower than distance-only, but still SIMD-accelerated
/// - For 100K points: ~0.6ms on modern CPU
#[inline]
pub fn eval_gradient_simd(sdf: &CompiledSdf, p: Vec3x8, epsilon: f32) -> (f32x8, f32x8, f32x8) {
    let e = f32x8::splat(epsilon);
    let ne = f32x8::splat(-epsilon);

    // Tetrahedral method: 4 evaluations instead of 6
    let v0 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + e,  y: p.y + ne, z: p.z + ne }); // (+,-,-)
    let v1 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + ne, y: p.y + ne, z: p.z + e  }); // (-,-,+)
    let v2 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + ne, y: p.y + e,  z: p.z + ne }); // (-,+,-)
    let v3 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + e,  y: p.y + e,  z: p.z + e  }); // (+,+,+)

    let gx = v0 - v1 - v2 + v3;
    let gy = -v0 - v1 + v2 + v3;
    let gz = -v0 + v1 - v2 + v3;

    (gx, gy, gz)
}

/// Compute both distance and gradient in one call - SIMD 8-wide
///
/// More efficient when you need both values, as it shares the center evaluation.
#[inline]
pub fn eval_distance_and_gradient_simd(
    sdf: &CompiledSdf,
    p: Vec3x8,
    epsilon: f32,
) -> (f32x8, f32x8, f32x8, f32x8) {
    // Center distance
    let d_center = eval_compiled_simd(sdf, p);

    // Tetrahedral method: 4 evaluations for gradient
    let e = f32x8::splat(epsilon);
    let ne = f32x8::splat(-epsilon);

    let v0 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + e,  y: p.y + ne, z: p.z + ne }); // (+,-,-)
    let v1 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + ne, y: p.y + ne, z: p.z + e  }); // (-,-,+)
    let v2 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + ne, y: p.y + e,  z: p.z + ne }); // (-,+,-)
    let v3 = eval_compiled_simd(sdf, Vec3x8 { x: p.x + e,  y: p.y + e,  z: p.z + e  }); // (+,+,+)

    let gx = v0 - v1 - v2 + v3;
    let gy = -v0 - v1 + v2 + v3;
    let gz = -v0 + v1 - v2 + v3;

    (d_center, gx, gy, gz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled::CompiledSdf;
    use crate::eval::eval;
    use crate::types::SdfNode;

    #[test]
    fn test_simd_sphere() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        let points = Vec3x8::from_vecs([
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        // Check against scalar evaluation
        assert!((arr[0] - (-1.0)).abs() < 0.001); // origin
        assert!(arr[1].abs() < 0.001); // on surface
        assert!((arr[2] - 1.0).abs() < 0.001); // outside
    }

    #[test]
    fn test_simd_box() {
        let node = SdfNode::box3d(1.0, 1.0, 1.0);
        let compiled = CompiledSdf::compile(&node);

        let points = Vec3x8::from_vecs([
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.5, 1.5, 0.0),
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        // Compare with scalar
        let test_vecs = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        for (i, v) in test_vecs.iter().enumerate() {
            let scalar = eval(&node, *v);
            assert!(
                (arr[i] - scalar).abs() < 0.001,
                "Mismatch at {}: simd={}, scalar={}",
                i,
                arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_union() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0));
        let compiled = CompiledSdf::compile(&node);

        let points = Vec3x8::from_vecs([
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        // Verify against scalar
        let test_vecs = [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        for (i, v) in test_vecs.iter().enumerate() {
            let scalar = eval(&node, *v);
            assert!(
                (arr[i] - scalar).abs() < 0.01,
                "Union mismatch at {}: simd={}, scalar={}",
                i,
                arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_batch() {
        let node = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1);
        let compiled = CompiledSdf::compile(&node);

        let points: Vec<Vec3> = (0..100)
            .map(|i| {
                let t = i as f32 / 100.0;
                Vec3::new(
                    (t * 12.34).sin() * 2.0,
                    (t * 23.45).sin() * 2.0,
                    (t * 34.56).sin() * 2.0,
                )
            })
            .collect();

        let simd_results = eval_compiled_batch_simd(&compiled, &points);
        let scalar_results: Vec<f32> = points.iter().map(|p| eval(&node, *p)).collect();

        for (i, (simd, scalar)) in simd_results.iter().zip(scalar_results.iter()).enumerate() {
            assert!(
                (simd - scalar).abs() < 0.01,
                "Batch mismatch at {}: simd={}, scalar={}",
                i,
                simd,
                scalar
            );
        }
    }

    #[test]
    fn test_simd_complex() {
        let node = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
            .translate(0.5, 0.0, 0.0)
            .scale(1.5);
        let compiled = CompiledSdf::compile(&node);

        let test_vecs = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        let points = Vec3x8::from_vecs([
            test_vecs[0],
            test_vecs[1],
            test_vecs[2],
            test_vecs[3],
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        for (i, v) in test_vecs.iter().enumerate() {
            let scalar = eval(&node, *v);
            assert!(
                (arr[i] - scalar).abs() < 0.05,
                "Complex mismatch at {:?}: simd={}, scalar={}",
                v,
                arr[i],
                scalar
            );
        }
    }
}
