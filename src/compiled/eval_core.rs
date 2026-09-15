//! The one bytecode stack machine, generic over [`Real`].
//!
//! `eval_compiled` (scalar), `eval_compiled_bvh` (BVH bytecode) and
//! `eval_compiled_simd` (8-lane) are all `eval_bytecode::<R>` for `R = f32` or
//! `R = wide::f32x8`. Transforms, modifiers, coordinate-frame handling and
//! post-processing are written exactly once here against the generic laws in
//! [`super::real`]; leaf primitives and CSG binary operators are dispatched
//! through [`PrimTable`] (per-instantiation bodies until Phase 2 of the 1.10
//! plan makes those generic as well).
//!
//! The `match` over [`OpCode`] is exhaustive on purpose: an opcode without an
//! arm here is a compile error, never a silent wrong distance.
//!
//! Author: Moroya Sakamoto

use super::instruction::Instruction;
use super::opcode::OpCode;
use super::prim_table::PrimTable;
use super::real::{self, Real, Vec3R};
use crate::modifiers::perlin_noise_3d;
use glam::{Quat, Vec2, Vec3};
use std::mem::MaybeUninit;

/// Maximum stack depth for value stack
pub(super) const MAX_VALUE_STACK: usize = 64;
/// Maximum stack depth for coordinate transforms
pub(super) const MAX_COORD_STACK: usize = 32;

/// Fixed-capacity, uninitialised slot array for the evaluator stacks.
///
/// The bytecode is a post-order stack program: a value is read at index `i`
/// only after `prim!` / `bin!` wrote it there, a frame at `csp` only after
/// `push_frame!` stored it, and the per-frame lane only after the pushing
/// opcode set it. Debug builds record every initialised slot and assert it
/// on read; release builds pay nothing for the discipline.
struct Slots<T: Copy, const N: usize> {
    buf: [MaybeUninit<T>; N],
    #[cfg(debug_assertions)]
    init: u64,
}

impl<T: Copy, const N: usize> Slots<T, N> {
    const _FITS_MASK: () = assert!(N <= 64, "debug init mask is a u64");

    #[inline(always)]
    const fn new() -> Self {
        let () = Self::_FITS_MASK;
        Self {
            buf: [const { MaybeUninit::uninit() }; N],
            #[cfg(debug_assertions)]
            init: 0,
        }
    }

    #[inline(always)]
    const fn set(&mut self, i: usize, v: T) {
        self.buf[i].write(v);
        #[cfg(debug_assertions)]
        {
            self.init |= 1u64 << i;
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> T {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.init & (1u64 << i) != 0,
            "evaluator stack slot {i} read before it was written"
        );
        // SAFETY: `set` wrote slot `i` before any read of it — the bytecode
        // is a stack program (see the type doc); debug builds assert it.
        unsafe { self.buf[i].assume_init() }
    }
}

/// Coordinate frame on the transform stack
#[derive(Clone, Copy)]
struct Frame<R: Real> {
    /// Point before the transform (restored at PopTransform)
    point: Vec3R<R>,
    /// Scale correction before the transform
    /// Opcode that pushed this frame (selects post-processing)
    opcode: OpCode,
    /// Scalar parameters for post-processing
    params: [f32; 4],
    /// Auxiliary data window
    aux_offset: u32,
    aux_len: u32,
}

/// Execute a compiled instruction stream at `point` and return the signed distance.
///
/// `aux_data` is the side buffer referenced by `Instruction::aux_offset` /
/// `aux_len` (heightmaps, lattices, skinning bones, IFS matrices, polygon
/// vertices).
#[inline]
pub(super) fn eval_bytecode<R: PrimTable>(
    instructions: &[Instruction],
    aux_data: &[f32],
    point: Vec3R<R>,
) -> R {
    // Uninitialised slots: zero-filling ~3.4 KB (f32) / ~6 KB (f32x8) of
    // stack per call cost ≈ 21 ns, which made `eval_compiled` slower than
    // the tree walker for anything under ~30 nodes (external review
    // 2026-09-15, SDF-R2-4). Every read is preceded by a write at the same
    // index by the stack discipline below; debug builds check it.
    let mut value_stack: Slots<R, MAX_VALUE_STACK> = Slots::new();
    let mut vsp: usize = 0;

    let mut coord_stack: Slots<Frame<R>, MAX_COORD_STACK> = Slots::new();
    // Per-frame lane value written only by the opcodes that need it at PopTransform:
    // Extrude (original z) and LatticeDeform (Jacobian correction). Kept out of
    // `Frame` so the common push stays as small as a pre-1.10 frame.
    let mut frame_lane: Slots<R, MAX_COORD_STACK> = Slots::new();
    let mut csp: usize = 0;

    let mut p = point;
    // Leaf laws still take a scale multiplier (always one since 1.11.0: the
    // scale is applied when the Scale frame pops, see PopTransform). Dropping
    // the parameter from the ~125 leaf laws is a separate mechanical change.
    let scale_correction = R::one();

    macro_rules! push_frame {
        ($inst:expr, $op:expr) => {{
            // Direct struct-literal store (no temporary): keeps the push at ~one
            // store per field like the pre-1.10 evaluators.
            coord_stack.set(
                csp,
                Frame {
                    point: p,
                    opcode: $op,
                    params: [
                        $inst.params[0],
                        $inst.params[1],
                        $inst.params[2],
                        $inst.params[3],
                    ],
                    aux_offset: $inst.aux_offset,
                    aux_len: $inst.aux_len,
                },
            );
            csp += 1;
        }};
    }
    macro_rules! prim {
        ($inst:expr, $law:ident) => {{
            value_stack.set(vsp, R::$law($inst, aux_data, p, scale_correction));
            vsp += 1;
        }};
    }
    macro_rules! bin {
        ($inst:expr, $law:ident) => {{
            vsp -= 1;
            let b = value_stack.get(vsp);
            let a = value_stack.get(vsp - 1);
            value_stack.set(vsp - 1, R::$law($inst, a, b));
        }};
    }

    for inst in instructions.iter() {
        let op = inst.opcode;
        match op {
            // === Transforms ===
            OpCode::Translate => {
                push_frame!(inst, op);
                p = p - Vec3R::splat(Vec3::new(inst.params[0], inst.params[1], inst.params[2]));
            }
            OpCode::Rotate => {
                push_frame!(inst, op);
                let q = Quat::from_xyzw(
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                p = real::rotate_inverse(q, p);
            }
            OpCode::Scale => {
                // params[0] = 1/factor, params[1] = factor
                // The distance is scaled when the frame pops (see PopTransform),
                // not at the leaves: `s * f(p / s)` is only equal to "scale every
                // primitive distance by s" when everything between the leaves and
                // this node is linear. Smooth / exp / chamfer / stairs blends, round,
                // onion and displacement all carry an absolute width and are not
                // (found by `fuzz_eval_parity`: Scale(ExpSmoothUnion) was 21% off).
                push_frame!(inst, op);
                p = p * R::splat(inst.params[0]);
            }
            OpCode::ScaleNonUniform => {
                // params[0..3] = 1/s, params[3] = min(sx, sy, sz)
                push_frame!(inst, op);
                p = p.mul_vec(Vec3R::splat(Vec3::new(
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                )));
                // Lipschitz-bound correction applied when the frame pops (see Scale)
            }
            OpCode::ProjectiveTransform => {
                push_frame!(inst, op);
                let aux_off = inst.aux_offset as usize;
                if inst.aux_len >= 16 {
                    let mut inv_m = [0.0f32; 16];
                    inv_m.copy_from_slice(&aux_data[aux_off..aux_off + 16]);
                    p = p.map(|q| crate::transforms::projective::projective_transform(q, &inv_m).0);
                }
            }
            OpCode::LatticeDeform => {
                push_frame!(inst, op);
                let aux_off = inst.aux_offset as usize;
                if inst.aux_len >= 9 {
                    let aux = &aux_data[aux_off..aux_off + inst.aux_len as usize];
                    let nx = aux[0] as u32;
                    let ny = aux[1] as u32;
                    let nz = aux[2] as u32;
                    let bbox_min = Vec3::new(aux[3], aux[4], aux[5]);
                    let bbox_max = Vec3::new(aux[6], aux[7], aux[8]);
                    let cp_data = &aux[9..];
                    let num_cps = cp_data.len() / 3;
                    let control_points: Vec<Vec3> = (0..num_cps)
                        .map(|i| Vec3::new(cp_data[i * 3], cp_data[i * 3 + 1], cp_data[i * 3 + 2]))
                        .collect();
                    let (q, correction) = R::map3vs(p.x, p.y, p.z, |q| {
                        crate::transforms::lattice::lattice_deform(
                            q,
                            &control_points,
                            nx,
                            ny,
                            nz,
                            bbox_min,
                            bbox_max,
                        )
                    });
                    p = q;
                    // Tree law: eval(child, q) / correction — applied at PopTransform
                    frame_lane.set(csp - 1, correction);
                } else {
                    frame_lane.set(csp - 1, R::one());
                }
            }
            OpCode::SdfSkinning => {
                push_frame!(inst, op);
                let aux_off = inst.aux_offset as usize;
                if inst.aux_len >= 1 {
                    let aux = &aux_data[aux_off..aux_off + inst.aux_len as usize];
                    let bone_count = aux[0] as usize;
                    let mut bones = Vec::with_capacity(bone_count);
                    let mut cursor = 1;
                    for _ in 0..bone_count {
                        let mut inv_bind = [0.0f32; 16];
                        let mut cur_pose = [0.0f32; 16];
                        inv_bind.copy_from_slice(&aux[cursor..cursor + 16]);
                        cursor += 16;
                        cur_pose.copy_from_slice(&aux[cursor..cursor + 16]);
                        cursor += 16;
                        let weight = aux[cursor];
                        cursor += 1;
                        bones.push(crate::transforms::skinning::BoneTransform {
                            inv_bind_pose: inv_bind,
                            current_pose: cur_pose,
                            weight,
                        });
                    }
                    p = p.map(|q| crate::transforms::skinning::sdf_skinning(q, &bones).0);
                }
            }

            // === Modifiers (point-modifying, prefix) ===
            OpCode::Twist => {
                push_frame!(inst, op);
                p = real::twist(p, inst.params[0]);
            }
            OpCode::Bend => {
                push_frame!(inst, op);
                p = real::bend(p, inst.params[0]);
            }
            OpCode::RepeatInfinite => {
                // params[0..3] = spacing, params[3..6] = 1/spacing
                push_frame!(inst, op);
                let spacing = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let recip = Vec3::new(inst.params[3], inst.params[4], inst.params[5]);
                p = real::repeat_infinite(p, spacing, recip);
            }
            OpCode::RepeatFinite => {
                // params[0..3] = counts, params[3..6] = spacing
                push_frame!(inst, op);
                let count = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let spacing = Vec3::new(inst.params[3], inst.params[4], inst.params[5]);
                p = real::repeat_finite(p, count, spacing);
            }
            OpCode::Elongate => {
                push_frame!(inst, op);
                p = real::elongate(p, Vec3::new(inst.params[0], inst.params[1], inst.params[2]));
            }
            OpCode::Mirror => {
                push_frame!(inst, op);
                p = real::mirror(p, Vec3::new(inst.params[0], inst.params[1], inst.params[2]));
            }
            OpCode::OctantMirror => {
                push_frame!(inst, op);
                p = real::octant_mirror(p);
            }
            OpCode::Revolution => {
                push_frame!(inst, op);
                p = real::revolution(p, inst.params[0]);
            }
            OpCode::Extrude => {
                push_frame!(inst, op);
                frame_lane.set(csp - 1, p.z); // original z for the post-process
                p = real::extrude_point(p);
            }
            OpCode::Taper => {
                push_frame!(inst, op);
                p = real::taper(p, inst.params[0]);
            }
            OpCode::PolarRepeat => {
                // params[1] = sector, params[2] = 1/sector
                push_frame!(inst, op);
                p = real::polar_repeat(p, inst.params[1], inst.params[2]);
            }
            OpCode::SweepBezier => {
                push_frame!(inst, op);
                p = real::sweep_bezier(
                    p,
                    Vec2::new(inst.params[0], inst.params[1]),
                    Vec2::new(inst.params[2], inst.params[3]),
                    Vec2::new(inst.params[4], inst.params[5]),
                );
            }
            OpCode::Shear => {
                push_frame!(inst, op);
                p = real::shear(p, Vec3::new(inst.params[0], inst.params[1], inst.params[2]));
            }
            OpCode::IcosahedralSymmetry => {
                push_frame!(inst, op);
                p = p.map(crate::modifiers::icosahedral_fold);
            }
            OpCode::IFS => {
                push_frame!(inst, op);
                let iterations = inst.params[0] as u32;
                let aux_off = inst.aux_offset as usize;
                if inst.aux_len >= 1 {
                    let aux = &aux_data[aux_off..aux_off + inst.aux_len as usize];
                    let transform_count = aux[0] as usize;
                    let mut transforms = Vec::with_capacity(transform_count);
                    for i in 0..transform_count {
                        let base = 1 + i * 16;
                        let mut mat = [0.0f32; 16];
                        mat.copy_from_slice(&aux[base..base + 16]);
                        transforms.push(mat);
                    }
                    p = p.map(|q| {
                        crate::modifiers::ifs_fold_with_scale(q, &transforms, iterations).0
                    });
                }
            }

            // === Modifiers (distance post-processing only, prefix) ===
            OpCode::Round
            | OpCode::Onion
            | OpCode::Noise
            | OpCode::Displacement
            | OpCode::HeightmapDisplacement
            | OpCode::SurfaceRoughness => {
                push_frame!(inst, op);
            }

            // === Animated (static evaluation: pass-through) ===
            // `CompiledSdf::compile` inlines `SdfNode::Animated` as its child and never
            // emits this opcode. If a bytecode producer does emit it, it must be paired
            // with `PopTransform`.
            OpCode::Animated => {
                push_frame!(inst, op);
            }

            // === Control ===
            OpCode::PopTransform => {
                csp -= 1;
                let frame = coord_stack.get(csp);

                // Post-processing keyed on the opcode that pushed the frame; point-only
                // transforms / modifiers fall through without touching the value stack.
                match frame.opcode {
                    OpCode::Scale => {
                        // params[1] = factor: d = factor * f(p / factor)
                        value_stack.set(
                            vsp - 1,
                            value_stack.get(vsp - 1) * R::splat(frame.params[1]),
                        );
                    }
                    OpCode::ScaleNonUniform => {
                        // params[3] = min(sx, sy, sz): conservative Lipschitz bound
                        value_stack.set(
                            vsp - 1,
                            value_stack.get(vsp - 1) * R::splat(frame.params[3]),
                        );
                    }
                    OpCode::Round => {
                        value_stack.set(
                            vsp - 1,
                            value_stack.get(vsp - 1) - R::splat(frame.params[0]),
                        );
                    }
                    OpCode::Onion => {
                        value_stack.set(
                            vsp - 1,
                            value_stack.get(vsp - 1).abs() - R::splat(frame.params[0]),
                        );
                    }
                    OpCode::Noise => {
                        let amplitude = frame.params[0];
                        let frequency = frame.params[1];
                        let seed = frame.params[2] as u32;
                        let n = frame.point.map_scalar(|q| {
                            let q = q * frequency;
                            perlin_noise_3d(q.x, q.y, q.z, seed)
                        });
                        value_stack
                            .set(vsp - 1, value_stack.get(vsp - 1) + n * R::splat(amplitude));
                    }
                    OpCode::Extrude => {
                        value_stack.set(
                            vsp - 1,
                            real::extrude_distance(
                                value_stack.get(vsp - 1),
                                frame_lane.get(csp),
                                frame.params[0],
                            ),
                        );
                    }
                    OpCode::Displacement => {
                        // amplitude + per-axis frequency (legacy Displacement = 5,5,5)
                        let amplitude = frame.params[0];
                        let frequency =
                            Vec3::new(frame.params[1], frame.params[2], frame.params[3]);
                        value_stack.set(
                            vsp - 1,
                            R::map_dp(value_stack.get(vsp - 1), frame.point, |d, q| {
                                crate::modifiers::modifier_sine_displacement(
                                    d, q, amplitude, frequency,
                                )
                            }),
                        );
                    }
                    OpCode::ProjectiveTransform => {
                        value_stack.set(
                            vsp - 1,
                            value_stack.get(vsp - 1) * R::splat(frame.params[0]),
                        );
                    }
                    OpCode::LatticeDeform => {
                        value_stack.set(vsp - 1, value_stack.get(vsp - 1) / frame_lane.get(csp));
                    }
                    OpCode::HeightmapDisplacement => {
                        let amplitude = frame.params[0];
                        let hm_scale = frame.params[1];
                        let aux_off = frame.aux_offset as usize;
                        if frame.aux_len >= 2 {
                            let aux = &aux_data[aux_off..aux_off + frame.aux_len as usize];
                            let w = aux[0] as u32;
                            let h = aux[1] as u32;
                            let hmap = &aux[2..];
                            let disp = frame.point.map_scalar(|q| {
                                crate::modifiers::heightmap_displacement(
                                    q, hmap, w, h, amplitude, hm_scale,
                                )
                            });
                            value_stack.set(vsp - 1, value_stack.get(vsp - 1) - disp);
                        }
                    }
                    OpCode::SurfaceRoughness => {
                        let frequency = frame.params[0];
                        let amplitude = frame.params[1];
                        let octaves = frame.params[2] as u32;
                        value_stack.set(
                            vsp - 1,
                            R::map_dp(value_stack.get(vsp - 1), frame.point, |d, q| {
                                crate::modifiers::surface_roughness(
                                    q, d, frequency, amplitude, octaves,
                                )
                            }),
                        );
                    }
                    // Point-only transforms / modifiers: nothing to post-process
                    _ => {}
                }

                // Restore coordinate state
                p = frame.point;
            }

            OpCode::End => break,

            // === Leaf primitives (one inlined law each, per instantiation) ===
            OpCode::Sphere => prim!(inst, sphere),
            OpCode::Box3d => prim!(inst, box3d),
            OpCode::Cylinder => prim!(inst, cylinder),
            OpCode::Torus => prim!(inst, torus),
            OpCode::Plane => prim!(inst, plane),
            OpCode::Capsule => prim!(inst, capsule),
            OpCode::Cone => prim!(inst, cone),
            OpCode::Ellipsoid => prim!(inst, ellipsoid),
            OpCode::RoundedCone => prim!(inst, rounded_cone),
            OpCode::Pyramid => prim!(inst, pyramid),
            OpCode::Octahedron => prim!(inst, octahedron),
            OpCode::HexPrism => prim!(inst, hex_prism),
            OpCode::Link => prim!(inst, link),
            OpCode::RoundedBox => prim!(inst, rounded_box),
            OpCode::CappedCone => prim!(inst, capped_cone),
            OpCode::CappedTorus => prim!(inst, capped_torus),
            OpCode::RoundedCylinder => prim!(inst, rounded_cylinder),
            OpCode::TriangularPrism => prim!(inst, triangular_prism),
            OpCode::CutSphere => prim!(inst, cut_sphere),
            OpCode::CutHollowSphere => prim!(inst, cut_hollow_sphere),
            OpCode::DeathStar => prim!(inst, death_star),
            OpCode::SolidAngle => prim!(inst, solid_angle),
            OpCode::Rhombus => prim!(inst, rhombus),
            OpCode::Horseshoe => prim!(inst, horseshoe),
            OpCode::Vesica => prim!(inst, vesica),
            OpCode::InfiniteCylinder => prim!(inst, infinite_cylinder),
            OpCode::InfiniteCone => prim!(inst, infinite_cone),
            OpCode::Gyroid => prim!(inst, gyroid),
            OpCode::Heart => prim!(inst, heart),
            OpCode::Tube => prim!(inst, tube),
            OpCode::Barrel => prim!(inst, barrel),
            OpCode::Diamond => prim!(inst, diamond),
            OpCode::ChamferedCube => prim!(inst, chamfered_cube),
            OpCode::SchwarzP => prim!(inst, schwarz_p),
            OpCode::Superellipsoid => prim!(inst, superellipsoid),
            OpCode::RoundedX => prim!(inst, rounded_x),
            OpCode::Pie => prim!(inst, pie),
            OpCode::Trapezoid => prim!(inst, trapezoid),
            OpCode::Parallelogram => prim!(inst, parallelogram),
            OpCode::Tunnel => prim!(inst, tunnel),
            OpCode::UnevenCapsule => prim!(inst, uneven_capsule),
            OpCode::Egg => prim!(inst, egg),
            OpCode::ArcShape => prim!(inst, arc_shape),
            OpCode::Moon => prim!(inst, moon),
            OpCode::CrossShape => prim!(inst, cross_shape),
            OpCode::BlobbyCross => prim!(inst, blobby_cross),
            OpCode::ParabolaSegment => prim!(inst, parabola_segment),
            OpCode::RegularPolygon => prim!(inst, regular_polygon),
            OpCode::StarPolygon => prim!(inst, star_polygon),
            OpCode::Stairs => prim!(inst, stairs),
            OpCode::Helix => prim!(inst, helix),
            OpCode::Tetrahedron => prim!(inst, tetrahedron),
            OpCode::Dodecahedron => prim!(inst, dodecahedron),
            OpCode::Icosahedron => prim!(inst, icosahedron),
            OpCode::TruncatedOctahedron => prim!(inst, truncated_octahedron),
            OpCode::TruncatedIcosahedron => prim!(inst, truncated_icosahedron),
            OpCode::BoxFrame => prim!(inst, box_frame),
            OpCode::DiamondSurface => prim!(inst, diamond_surface),
            OpCode::Neovius => prim!(inst, neovius),
            OpCode::Lidinoid => prim!(inst, lidinoid),
            OpCode::IWP => prim!(inst, iwp),
            OpCode::FRD => prim!(inst, frd),
            OpCode::FischerKochS => prim!(inst, fischer_koch_s),
            OpCode::PMY => prim!(inst, pmy),
            OpCode::Circle2D => prim!(inst, circle_2d),
            OpCode::Rect2D => prim!(inst, rect_2d),
            OpCode::Segment2D => prim!(inst, segment_2d),
            OpCode::Polygon2D => prim!(inst, polygon_2d),
            OpCode::RoundedRect2D => prim!(inst, rounded_rect_2d),
            OpCode::Annular2D => prim!(inst, annular_2d),

            // === CSG binary operators ===
            OpCode::Union => bin!(inst, union),
            OpCode::Intersection => bin!(inst, intersection),
            OpCode::Subtraction => bin!(inst, subtraction),
            OpCode::SmoothUnion => bin!(inst, smooth_union),
            OpCode::SmoothIntersection => bin!(inst, smooth_intersection),
            OpCode::SmoothSubtraction => bin!(inst, smooth_subtraction),
            OpCode::ChamferUnion => bin!(inst, chamfer_union),
            OpCode::ChamferIntersection => bin!(inst, chamfer_intersection),
            OpCode::ChamferSubtraction => bin!(inst, chamfer_subtraction),
            OpCode::StairsUnion => bin!(inst, stairs_union),
            OpCode::StairsIntersection => bin!(inst, stairs_intersection),
            OpCode::StairsSubtraction => bin!(inst, stairs_subtraction),
            OpCode::XOR => bin!(inst, xor),
            OpCode::Morph => bin!(inst, morph),
            OpCode::ColumnsUnion => bin!(inst, columns_union),
            OpCode::ColumnsIntersection => bin!(inst, columns_intersection),
            OpCode::ColumnsSubtraction => bin!(inst, columns_subtraction),
            OpCode::Pipe => bin!(inst, pipe),
            OpCode::Engrave => bin!(inst, engrave),
            OpCode::Groove => bin!(inst, groove),
            OpCode::Tongue => bin!(inst, tongue),
            OpCode::ExpSmoothUnion => bin!(inst, exp_smooth_union),
            OpCode::ExpSmoothIntersection => bin!(inst, exp_smooth_intersection),
            OpCode::ExpSmoothSubtraction => bin!(inst, exp_smooth_subtraction),
            // NOTE: no `_ =>` arm. Every OpCode variant must be handled above.
        }
    }

    if vsp > 0 {
        value_stack.get(0)
    } else {
        R::splat(f32::MAX)
    }
}
