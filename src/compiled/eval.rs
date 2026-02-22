//! Stack-based SDF evaluation for compiled bytecode
//!
//! This evaluator uses a simple stack machine instead of recursion,
//! providing better cache locality and avoiding function call overhead.
//!
//! Author: Moroya Sakamoto

use super::compiler::CompiledSdf;
use super::opcode::OpCode;
use crate::modifiers::*;
use crate::operations::*;
use crate::primitives::*;
use glam::{Quat, Vec2, Vec3};

use crate::modifiers::perlin_noise_3d;

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
    #[allow(dead_code)]
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
                let radius = inst.get_capsule_radius();
                let d = sdf_capsule(p, point_a, point_b, radius);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Cone => {
                let d = sdf_cone(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Ellipsoid => {
                let radii = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_ellipsoid(p, radii);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::RoundedCone => {
                let d = sdf_rounded_cone(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Pyramid => {
                let d = sdf_pyramid(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Octahedron => {
                let d = sdf_octahedron(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::HexPrism => {
                let d = sdf_hex_prism(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Link => {
                let d = sdf_link(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            // === Extended Primitives ===
            OpCode::RoundedBox => {
                let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_rounded_box(p, half_extents, inst.params[3]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::CappedCone => {
                let d = sdf_capped_cone(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::CappedTorus => {
                let d = sdf_capped_torus(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::RoundedCylinder => {
                let d = sdf_rounded_cylinder(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::TriangularPrism => {
                let d = sdf_triangular_prism(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::CutSphere => {
                let d = sdf_cut_sphere(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::CutHollowSphere => {
                let d = sdf_cut_hollow_sphere(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::DeathStar => {
                let d = sdf_death_star(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::SolidAngle => {
                let d = sdf_solid_angle(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Rhombus => {
                let d = sdf_rhombus(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Horseshoe => {
                let d = sdf_horseshoe(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                    inst.params[4],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Vesica => {
                let d = sdf_vesica(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::InfiniteCylinder => {
                let d = sdf_infinite_cylinder(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::InfiniteCone => {
                let d = sdf_infinite_cone(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Gyroid => {
                let d = sdf_gyroid(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Heart => {
                let d = sdf_heart(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Tube => {
                let d = sdf_tube(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Barrel => {
                let d = sdf_barrel(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Diamond => {
                let d = sdf_diamond(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::ChamferedCube => {
                let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_chamfered_cube(p, half_extents, inst.params[3]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::SchwarzP => {
                let d = sdf_schwarz_p(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Superellipsoid => {
                let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_superellipsoid(p, half_extents, inst.params[3], inst.params[4]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::RoundedX => {
                let d = sdf_rounded_x(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Pie => {
                let d = sdf_pie(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Trapezoid => {
                let d = sdf_trapezoid(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Parallelogram => {
                let d = sdf_parallelogram(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Tunnel => {
                let d = sdf_tunnel(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::UnevenCapsule => {
                let d = sdf_uneven_capsule(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Egg => {
                let d = sdf_egg(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::ArcShape => {
                let d = sdf_arc_shape(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Moon => {
                let d = sdf_moon(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::CrossShape => {
                let d = sdf_cross_shape(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::BlobbyCross => {
                let d = sdf_blobby_cross(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::ParabolaSegment => {
                let d = sdf_parabola_segment(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::RegularPolygon => {
                let d = sdf_regular_polygon(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::StarPolygon => {
                let d = sdf_star_polygon(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Stairs => {
                let d = sdf_stairs(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Helix => {
                let d = sdf_helix(
                    p,
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Tetrahedron => {
                let d = sdf_tetrahedron(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Dodecahedron => {
                let d = sdf_dodecahedron(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Icosahedron => {
                let d = sdf_icosahedron(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::TruncatedOctahedron => {
                let d = sdf_truncated_octahedron(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::TruncatedIcosahedron => {
                let d = sdf_truncated_icosahedron(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::BoxFrame => {
                let he = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_box_frame(p, he, inst.params[3]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::DiamondSurface => {
                let d = sdf_diamond_surface(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Neovius => {
                let d = sdf_neovius(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Lidinoid => {
                let d = sdf_lidinoid(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::IWP => {
                let d = sdf_iwp(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::FRD => {
                let d = sdf_frd(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::FischerKochS => {
                let d = sdf_fischer_koch_s(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::PMY => {
                let d = sdf_pmy(p, inst.params[0], inst.params[1]);
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
                // Division Exorcism: params[1] = precomputed 1/k
                value_stack[vsp - 1] = sdf_smooth_union_rk(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::SmoothIntersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] =
                    sdf_smooth_intersection_rk(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::SmoothSubtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] =
                    sdf_smooth_subtraction_rk(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::ChamferUnion => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_chamfer_union(a, b, inst.params[0]);
            }

            OpCode::ChamferIntersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_chamfer_intersection(a, b, inst.params[0]);
            }

            OpCode::ChamferSubtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_chamfer_subtraction(a, b, inst.params[0]);
            }

            OpCode::StairsUnion => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_stairs_union(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::StairsIntersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] =
                    sdf_stairs_intersection(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::StairsSubtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_stairs_subtraction(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::XOR => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_xor(a, b);
            }

            OpCode::Morph => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_morph(a, b, inst.params[0]);
            }

            OpCode::ColumnsUnion => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_columns_union(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::ColumnsIntersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] =
                    sdf_columns_intersection(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::ColumnsSubtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] =
                    sdf_columns_subtraction(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::Pipe => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_pipe(a, b, inst.params[0]);
            }

            OpCode::Engrave => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_engrave(a, b, inst.params[0]);
            }

            OpCode::Groove => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_groove(a, b, inst.params[0], inst.params[1]);
            }

            OpCode::Tongue => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_tongue(a, b, inst.params[0], inst.params[1]);
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
                p -= offset;
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
                let inv_factor = inst.params[0]; // precomputed 1.0/factor
                let factor = inst.params[1]; // original factor
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Scale,
                    params: [factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                // Multiply by precomputed inverse (no division)
                p *= inv_factor;
                scale_correction *= factor;
            }

            OpCode::ScaleNonUniform => {
                let inv_factors = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let min_factor = inst.params[3]; // precomputed min(sx,sy,sz)

                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::ScaleNonUniform,
                    params: [min_factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                // Multiply by precomputed inverses (no division)
                p *= inv_factors;
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

                // Division Exorcism: params[3..5] = precomputed 1/spacing
                let spacing = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let recip = Vec3::new(inst.params[3], inst.params[4], inst.params[5]);
                p = modifier_repeat_infinite_rk(p, spacing, recip);
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

            OpCode::Mirror => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Mirror,
                    params: [0.0; 4],
                };
                csp += 1;

                let axes = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = modifier_mirror(p, axes);
            }

            OpCode::OctantMirror => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::OctantMirror,
                    params: [0.0; 4],
                };
                csp += 1;

                // abs + sort so x >= y >= z
                let mut x = p.x.abs();
                let mut y = p.y.abs();
                let mut z = p.z.abs();
                if y > x {
                    std::mem::swap(&mut x, &mut y);
                }
                if z > y {
                    std::mem::swap(&mut y, &mut z);
                }
                if y > x {
                    std::mem::swap(&mut x, &mut y);
                }
                p = Vec3::new(x, y, z);
            }

            OpCode::Revolution => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Revolution,
                    params: [0.0; 4],
                };
                csp += 1;

                p = modifier_revolution(p, inst.params[0]);
            }

            OpCode::Extrude => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Extrude,
                    params: [inst.params[0], p.z, 0.0, 0.0], // store half_height and original z
                };
                csp += 1;

                // Evaluate child in XY plane
                p = modifier_extrude_point(p);
            }

            OpCode::Taper => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Taper,
                    params: [0.0; 4],
                };
                csp += 1;

                p = modifier_taper(p, inst.params[0]);
            }

            OpCode::Displacement => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::Displacement,
                    params: [inst.params[0], 0.0, 0.0, 0.0], // store strength
                };
                csp += 1;
                // Displacement doesn't modify point, only post-processes distance
            }

            OpCode::PolarRepeat => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::PolarRepeat,
                    params: [0.0; 4],
                };
                csp += 1;

                // Division Exorcism: params[1]=sector, params[2]=recip_sector
                p = modifier_polar_repeat_rk(p, inst.params[1], inst.params[2]);
            }

            OpCode::SweepBezier => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    inst_idx,
                    opcode: OpCode::SweepBezier,
                    params: [0.0; 4],
                };
                csp += 1;

                let p0 = Vec2::new(inst.params[0], inst.params[1]);
                let p1 = Vec2::new(inst.params[2], inst.params[3]);
                let p2 = Vec2::new(inst.params[4], inst.params[5]);
                p = modifier_sweep_bezier(p, p0, p1, p2);
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
                        let p = frame.point * frequency;
                        let noise_val = perlin_noise_3d(p.x, p.y, p.z, seed);
                        value_stack[vsp - 1] += noise_val * amplitude;
                    }
                    OpCode::Extrude => {
                        let half_height = frame.params[0];
                        let original_z = frame.params[1];
                        let d = value_stack[vsp - 1];
                        value_stack[vsp - 1] = modifier_extrude(d, original_z, half_height);
                    }
                    OpCode::Displacement => {
                        let strength = frame.params[0];
                        let d = value_stack[vsp - 1];
                        value_stack[vsp - 1] = modifier_displacement(d, frame.point, strength);
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

            // New opcodes: fallback to zero (these need full implementations)
            _ => {
                // 2D prims and new ops push a value
                if inst.opcode.is_primitive() {
                    value_stack[vsp] = p.length() * scale_correction;
                    vsp += 1;
                } else if inst.opcode.is_binary_op() {
                    vsp -= 1;
                    // Leave top of stack unchanged
                }
                // Modifiers: do nothing (child will handle)
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
    let e = epsilon;

    // Tetrahedral method: 4 evaluations instead of 6
    let v0 = eval_compiled(sdf, point + Vec3::new(e, -e, -e)); // (+,-,-)
    let v1 = eval_compiled(sdf, point + Vec3::new(-e, -e, e)); // (-,-,+)
    let v2 = eval_compiled(sdf, point + Vec3::new(-e, e, -e)); // (-,+,-)
    let v3 = eval_compiled(sdf, point + Vec3::new(e, e, e)); // (+,+,+)

    Vec3::new(v0 - v1 - v2 + v3, -v0 - v1 + v2 + v3, -v0 + v1 - v2 + v3).normalize()
}

/// Combined distance + normal from 4 evaluations (tetrahedral method).
///
/// The distance at center is approximated as the average of the 4 tetrahedral
/// offset distances. This avoids the 5th eval needed when calling
/// `eval_compiled` + `eval_compiled_normal` separately.
///
/// Accuracy: distance error ≈ O(epsilon²), negligible for collision detection.
pub fn eval_compiled_distance_and_normal(
    sdf: &CompiledSdf,
    point: Vec3,
    epsilon: f32,
) -> (f32, Vec3) {
    let e = epsilon;

    let v0 = eval_compiled(sdf, point + Vec3::new(e, -e, -e));
    let v1 = eval_compiled(sdf, point + Vec3::new(-e, -e, e));
    let v2 = eval_compiled(sdf, point + Vec3::new(-e, e, -e));
    let v3 = eval_compiled(sdf, point + Vec3::new(e, e, e));

    // Distance ≈ average of the 4 offset samples
    let dist = (v0 + v1 + v2 + v3) * 0.25;

    let normal = Vec3::new(v0 - v1 - v2 + v3, -v0 - v1 + v2 + v3, -v0 + v1 - v2 + v3).normalize();

    (dist, normal)
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
    use crate::eval::eval;
    use crate::types::SdfNode;

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
            d_interpreted,
            d_compiled
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
                p,
                d_interpreted,
                d_compiled
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
                p,
                d_base,
                d_noise,
                diff
            );
        }

        // Verify noise is deterministic (same seed gives same result)
        let p = Vec3::new(0.7, 0.3, 0.5);
        let d1 = eval_compiled(&compiled_noise, p);
        let d2 = eval_compiled(&compiled_noise, p);
        assert_eq!(d1, d2, "Noise should be deterministic");
    }
}
