//! Stack-based SDF evaluation for compiled bytecode
//!
//! This evaluator uses a simple stack machine instead of recursion,
//! providing better cache locality and avoiding function call overhead.
//!
//! Author: Moroya Sakamoto

use super::compiler::CompiledSdf;
use super::opcode::OpCode;
use crate::primitives::*;
use crate::operations::*;
use crate::modifiers::*;
use glam::{Quat, Vec3};

/// Simple hash-based 3D noise for compiled SDF evaluation
///
/// Uses a hash function to generate pseudo-random values based on position.
/// Returns values in range [-1, 1].
#[inline]
fn hash_noise_3d(p: Vec3, seed: u32) -> f32 {
    // Hash function based on integer coordinates
    let ix = (p.x.floor() as i32) as u32;
    let iy = (p.y.floor() as i32) as u32;
    let iz = (p.z.floor() as i32) as u32;

    // Fractional parts for interpolation
    let fx = p.x.fract();
    let fy = p.y.fract();
    let fz = p.z.fract();

    // Smooth interpolation curves
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    // Hash at 8 corners
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

    // Trilinear interpolation
    let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

    let c00 = lerp(c000, c100, ux);
    let c10 = lerp(c010, c110, ux);
    let c01 = lerp(c001, c101, ux);
    let c11 = lerp(c011, c111, ux);

    let c0 = lerp(c00, c10, uy);
    let c1 = lerp(c01, c11, uy);

    lerp(c0, c1, uz)
}

/// Maximum stack depth for value stack
const MAX_VALUE_STACK: usize = 64;
/// Maximum stack depth for coordinate transforms
const MAX_COORD_STACK: usize = 32;

/// Coordinate frame on the transform stack
#[derive(Clone, Copy)]
struct CoordFrame {
    /// Original point (before transform)
    point: Vec3,
    /// Scale correction factor (for uniform scale)
    scale_correction: f32,
    /// Instruction index that pushed this frame
    inst_idx: usize,
    /// OpCode that pushed this frame (for post-processing)
    opcode: OpCode,
    /// Parameters for post-processing
    params: [f32; 4],
}

impl Default for CoordFrame {
    fn default() -> Self {
        CoordFrame {
            point: Vec3::ZERO,
            scale_correction: 1.0,
            inst_idx: 0,
            opcode: OpCode::End,
            params: [0.0; 4],
        }
    }
}

/// Evaluate a compiled SDF at a point
///
/// This is the main entry point for compiled SDF evaluation.
/// It uses a stack-based approach instead of recursion.
#[inline]
pub fn eval_compiled(sdf: &CompiledSdf, point: Vec3) -> f32 {
    // Value stack (for intermediate SDF distances)
    let mut value_stack: [f32; MAX_VALUE_STACK] = [0.0; MAX_VALUE_STACK];
    let mut vsp: usize = 0;

    // Coordinate transform stack
    let mut coord_stack: [CoordFrame; MAX_COORD_STACK] = [CoordFrame::default(); MAX_COORD_STACK];
    let mut csp: usize = 0;

    // Current evaluation point
    let mut p = point;
    // Current scale correction (accumulated from Scale transforms)
    let mut scale_correction: f32 = 1.0;

    for (inst_idx, inst) in sdf.instructions.iter().enumerate() {
        match inst.opcode {
            // === Primitives ===
            OpCode::Sphere => {
                let d = sdf_sphere(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Box3d => {
                let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_box3d(p, half_extents);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Cylinder => {
                let d = sdf_cylinder(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Torus => {
                let d = sdf_torus(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Plane => {
                let normal = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_plane(p, normal, inst.params[3]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Capsule => {
                let point_a = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let point_b = Vec3::new(inst.params[3], inst.params[4], inst.params[5]);
                // Radius is stored in skip_offset as f32 bits
                let radius = f32::from_bits(inst.skip_offset);
                let d = sdf_capsule(p, point_a, point_b, radius);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            // === Binary Operations ===
            OpCode::Union => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_union(a, b);
            }

            OpCode::Intersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_intersection(a, b);
            }

            OpCode::Subtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_subtraction(a, b);
            }

            OpCode::SmoothUnion => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_smooth_union(a, b, inst.params[0]);
            }

            OpCode::SmoothIntersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_smooth_intersection(a, b, inst.params[0]);
            }

            OpCode::SmoothSubtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_smooth_subtraction(a, b, inst.params[0]);
            }

            // === Transforms ===
            OpCode::Translate => {
                // Push current state
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Translate,
                    params: [0.0; 4],
                };
                csp += 1;

                // Apply inverse transform (subtract offset)
                let offset = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = p - offset;
            }

            OpCode::Rotate => {
                // Push current state
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Rotate,
                    params: [0.0; 4],
                };
                csp += 1;

                // Apply inverse transform (inverse quaternion rotation)
                let q = Quat::from_xyzw(
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                p = q.inverse() * p;
            }

            OpCode::Scale => {
                let factor = inst.params[0];
                // Push current state with post-process info
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Scale,
                    params: [factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                // Apply inverse transform (divide by scale)
                p = p / factor;
                scale_correction *= factor;
            }

            OpCode::ScaleNonUniform => {
                let factors = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let min_factor = factors.x.min(factors.y).min(factors.z);

                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::ScaleNonUniform,
                    params: [min_factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                p = p / factors;
                scale_correction *= min_factor;
            }

            // === Modifiers ===
            OpCode::Twist => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Twist,
                    params: [0.0; 4],
                };
                csp += 1;

                p = modifier_twist(p, inst.params[0]);
            }

            OpCode::Bend => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Bend,
                    params: [0.0; 4],
                };
                csp += 1;

                p = modifier_bend(p, inst.params[0]);
            }

            OpCode::RepeatInfinite => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::RepeatInfinite,
                    params: [0.0; 4],
                };
                csp += 1;

                let spacing = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = modifier_repeat_infinite(p, spacing);
            }

            OpCode::RepeatFinite => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::RepeatFinite,
                    params: [0.0; 4],
                };
                csp += 1;

                let count = [
                    inst.params[0] as u32,
                    inst.params[1] as u32,
                    inst.params[2] as u32,
                ];
                let spacing = Vec3::new(inst.params[3], inst.params[4], inst.params[5]);
                p = modifier_repeat_finite(p, count, spacing);
            }

            OpCode::Round => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Round,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;
                // Round doesn't modify point, only post-processes distance
            }

            OpCode::Onion => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Onion,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;
                // Onion doesn't modify point, only post-processes distance
            }

            OpCode::Elongate => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Elongate,
                    params: [0.0; 4],
                };
                csp += 1;

                let amount = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = p - p.clamp(-amount, amount);
            }

            OpCode::Noise => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
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

                // Apply post-processing based on what transform was pushed
                match frame.opcode {
                    OpCode::Round => {
                        // d = d - radius
                        value_stack[vsp - 1] -= frame.params[0];
                    }
                    OpCode::Onion => {
                        // d = |d| - thickness
                        value_stack[vsp - 1] = value_stack[vsp - 1].abs() - frame.params[0];
                    }
                    OpCode::Noise => {
                        // d += noise(p * frequency) * amplitude
                        let amplitude = frame.params[0];
                        let frequency = frame.params[1];
                        let seed = frame.params[2] as u32;
                        let noise_val = hash_noise_3d(frame.point * frequency, seed);
                        value_stack[vsp - 1] += noise_val * amplitude;
                    }
                    _ => {}
                }

                // Restore coordinate state
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
        f32::MAX
    }
}

/// Evaluate compiled SDF and compute normal using finite differences
#[inline]
pub fn eval_compiled_normal(sdf: &CompiledSdf, point: Vec3, epsilon: f32) -> Vec3 {
    let dx = Vec3::new(epsilon, 0.0, 0.0);
    let dy = Vec3::new(0.0, epsilon, 0.0);
    let dz = Vec3::new(0.0, 0.0, epsilon);

    Vec3::new(
        eval_compiled(sdf, point + dx) - eval_compiled(sdf, point - dx),
        eval_compiled(sdf, point + dy) - eval_compiled(sdf, point - dy),
        eval_compiled(sdf, point + dz) - eval_compiled(sdf, point - dz),
    )
    .normalize()
}

/// Batch evaluate compiled SDF at multiple points
pub fn eval_compiled_batch(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    points.iter().map(|p| eval_compiled(sdf, *p)).collect()
}

/// Parallel batch evaluate compiled SDF at multiple points
pub fn eval_compiled_batch_parallel(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    use rayon::prelude::*;
    points.par_iter().map(|p| eval_compiled(sdf, *p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use crate::eval::eval;

    #[test]
    fn test_eval_sphere() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        // At origin: distance = -1 (inside)
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.0001);

        // On surface: distance = 0
        let d = eval_compiled(&compiled, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.0001);

        // Outside: distance = 1
        let d = eval_compiled(&compiled, Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_box() {
        let node = SdfNode::box3d(1.0, 1.0, 1.0);
        let compiled = CompiledSdf::compile(&node);

        // At origin: inside
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!(d < 0.0);

        // On surface - compare with interpreted
        let p = Vec3::new(1.0, 0.0, 0.0);
        let d_interpreted = eval(&node, p);
        let d_compiled = eval_compiled(&compiled, p);
        assert!(
            (d_interpreted - d_compiled).abs() < 0.0001,
            "Box at (1,0,0): interpreted={}, compiled={}",
            d_interpreted, d_compiled
        );
    }

    #[test]
    fn test_eval_union() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0));
        let compiled = CompiledSdf::compile(&node);

        // At origin: inside first sphere
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.0001);

        // At (3, 0, 0): inside second sphere
        let d = eval_compiled(&compiled, Vec3::new(3.0, 0.0, 0.0));
        assert!((d + 1.0).abs() < 0.0001);

        // Between spheres: outside
        let d = eval_compiled(&compiled, Vec3::new(1.5, 0.0, 0.0));
        assert!(d > 0.0);
    }

    #[test]
    fn test_eval_translate() {
        let node = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
        let compiled = CompiledSdf::compile(&node);

        // At (2, 0, 0): center of translated sphere
        let d = eval_compiled(&compiled, Vec3::new(2.0, 0.0, 0.0));
        assert!((d + 1.0).abs() < 0.0001);

        // At origin: outside
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_scale() {
        let node = SdfNode::sphere(1.0).scale(2.0);
        let compiled = CompiledSdf::compile(&node);

        // At (2, 0, 0): on surface of scaled sphere
        let d = eval_compiled(&compiled, Vec3::new(2.0, 0.0, 0.0));
        assert!(d.abs() < 0.01);
    }

    #[test]
    fn test_compare_with_interpreted() {
        // Compare compiled vs interpreted results
        let node = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.0, 0.0, 0.0), 0.2);
        let compiled = CompiledSdf::compile(&node);

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        for p in test_points {
            let d_interpreted = eval(&node, p);
            let d_compiled = eval_compiled(&compiled, p);
            assert!(
                (d_interpreted - d_compiled).abs() < 0.001,
                "Mismatch at {:?}: interpreted={}, compiled={}",
                p, d_interpreted, d_compiled
            );
        }
    }

    #[test]
    fn test_eval_normal() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        let n = eval_compiled_normal(&compiled, Vec3::new(1.0, 0.0, 0.0), 0.001);
        assert!((n.x - 1.0).abs() < 0.01);
        assert!(n.y.abs() < 0.01);
        assert!(n.z.abs() < 0.01);
    }

    #[test]
    fn test_eval_noise() {
        let base_node = SdfNode::sphere(1.0);
        let noise_node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);

        let compiled_base = CompiledSdf::compile(&base_node);
        let compiled_noise = CompiledSdf::compile(&noise_node);

        // Test at several points
        let test_points = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        for p in test_points {
            let d_base = eval_compiled(&compiled_base, p);
            let d_noise = eval_compiled(&compiled_noise, p);

            // Noise should modify the distance by at most amplitude (0.1)
            let diff = (d_noise - d_base).abs();
            assert!(
                diff <= 0.1 + 0.001, // tolerance for floating point
                "Noise effect too large at {:?}: base={}, noise={}, diff={}",
                p, d_base, d_noise, diff
            );
        }

        // Verify noise is deterministic (same seed gives same result)
        let p = Vec3::new(0.7, 0.3, 0.5);
        let d1 = eval_compiled(&compiled_noise, p);
        let d2 = eval_compiled(&compiled_noise, p);
        assert_eq!(d1, d2, "Noise should be deterministic");
    }
}
