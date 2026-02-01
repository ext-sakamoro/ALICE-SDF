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
use wide::f32x8;

/// Scalar hash-based 3D noise for SIMD lane processing
#[inline]
fn hash_noise_3d_scalar(p: Vec3, seed: u32) -> f32 {
    let ix = (p.x.floor() as i32) as u32;
    let iy = (p.y.floor() as i32) as u32;
    let iz = (p.z.floor() as i32) as u32;

    let fx = p.x.fract();
    let fy = p.y.fract();
    let fz = p.z.fract();

    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    let hash = |x: u32, y: u32, z: u32| -> f32 {
        let mut h = x.wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(z.wrapping_mul(1274126177))
            .wrapping_add(seed);
        h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        h = h ^ (h >> 16);
        (h as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let c000 = hash(ix, iy, iz);
    let c100 = hash(ix.wrapping_add(1), iy, iz);
    let c010 = hash(ix, iy.wrapping_add(1), iz);
    let c110 = hash(ix.wrapping_add(1), iy.wrapping_add(1), iz);
    let c001 = hash(ix, iy, iz.wrapping_add(1));
    let c101 = hash(ix.wrapping_add(1), iy, iz.wrapping_add(1));
    let c011 = hash(ix, iy.wrapping_add(1), iz.wrapping_add(1));
    let c111 = hash(ix.wrapping_add(1), iy.wrapping_add(1), iz.wrapping_add(1));

    let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

    let c00 = lerp(c000, c100, ux);
    let c10 = lerp(c010, c110, ux);
    let c01 = lerp(c001, c101, ux);
    let c11 = lerp(c011, c111, ux);

    let c0 = lerp(c00, c10, uy);
    let c1 = lerp(c01, c11, uy);

    lerp(c0, c1, uz)
}

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
                let radius = f32x8::splat(f32::from_bits(inst.skip_offset));

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
                let h = (dot_pa_ba / dot_ba_ba).max(f32x8::ZERO).min(f32x8::ONE);

                // length(pa - ba * h) - radius
                let dx = pax - bax * h;
                let dy = pay - bay * h;
                let dz = paz - baz * h;
                let d = (dx * dx + dy * dy + dz * dz).sqrt() - radius;

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
                let factor = inst.params[0];
                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Scale,
                    params: [factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                let inv_factor = f32x8::splat(1.0 / factor);
                p = p * inv_factor;
                scale_correction = scale_correction * f32x8::splat(factor);
            }

            OpCode::ScaleNonUniform => {
                let sx = inst.params[0];
                let sy = inst.params[1];
                let sz = inst.params[2];
                let min_factor = sx.min(sy).min(sz);

                coord_stack[csp] = CoordFrameSimd {
                    point: p,
                    scale_correction,
                    opcode: OpCode::ScaleNonUniform,
                    params: [min_factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                p = Vec3x8 {
                    x: p.x / f32x8::splat(sx),
                    y: p.y / f32x8::splat(sy),
                    z: p.z / f32x8::splat(sz),
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
                let half = f32x8::splat(0.5);

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
                            noise_vals[i] = hash_noise_3d_scalar(pt, seed) * amplitude;
                        }
                        value_stack[vsp - 1] = value_stack[vsp - 1] + f32x8::new(noise_vals);
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

/// Smooth minimum (polynomial) for SIMD
#[inline]
fn smooth_min_simd(a: f32x8, b: f32x8, k: f32x8) -> f32x8 {
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
    let eps = f32x8::splat(epsilon);
    let zero = f32x8::ZERO;

    // Offset vectors for central differences
    let dx = Vec3x8 { x: eps, y: zero, z: zero };
    let dy = Vec3x8 { x: zero, y: eps, z: zero };
    let dz = Vec3x8 { x: zero, y: zero, z: eps };

    // 6 SDF evaluations using SIMD
    let d_x_pos = eval_compiled_simd(sdf, p + dx);
    let d_x_neg = eval_compiled_simd(sdf, p - dx);
    let d_y_pos = eval_compiled_simd(sdf, p + dy);
    let d_y_neg = eval_compiled_simd(sdf, p - dy);
    let d_z_pos = eval_compiled_simd(sdf, p + dz);
    let d_z_neg = eval_compiled_simd(sdf, p - dz);

    // Central difference gradient
    let gx = d_x_pos - d_x_neg;
    let gy = d_y_pos - d_y_neg;
    let gz = d_z_pos - d_z_neg;

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
    let eps = f32x8::splat(epsilon);
    let zero = f32x8::ZERO;

    // Center distance
    let d_center = eval_compiled_simd(sdf, p);

    // Offset vectors
    let dx = Vec3x8 { x: eps, y: zero, z: zero };
    let dy = Vec3x8 { x: zero, y: eps, z: zero };
    let dz = Vec3x8 { x: zero, y: zero, z: eps };

    // Forward differences (3 evaluations instead of 6)
    let d_x_pos = eval_compiled_simd(sdf, p + dx);
    let d_y_pos = eval_compiled_simd(sdf, p + dy);
    let d_z_pos = eval_compiled_simd(sdf, p + dz);

    // Forward difference gradient (less accurate but faster)
    let gx = d_x_pos - d_center;
    let gy = d_y_pos - d_center;
    let gz = d_z_pos - d_center;

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
