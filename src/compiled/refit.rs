//! Bytecode-driven BVH AABB refit.
//!
//! Case B P1 (bytecode-driven full refit): recomputes every AABB in a
//! [`crate::compiled::CompiledSdfBvh`] from the current
//! `Instruction.params[]` values, without needing the original `SdfNode`.
//!
//! Case B P2 (partial refit): given a `dirty` set of instruction indices,
//! walks the ancestor chain via `parent_indices` and only recomputes AABBs
//! for instructions that (a) are dirty themselves or (b) are ancestors of a
//! dirty instruction. Non-affected subtrees keep their cached AABBs.
//!
//! ## Opcode coverage (P1 + P3, ~54)
//!
//! Primitives (13): `Sphere`, `Box3d`, `Cylinder`, `Torus`, `Plane`,
//! `Capsule`, `Cone`, `Ellipsoid`, `RoundedCone`, `Pyramid`, `Octahedron`,
//! `HexPrism`, `Link`.
//!
//! CSG binary (23): `Union`, `Intersection`, `Subtraction`, `SmoothUnion`,
//! `SmoothIntersection`, `SmoothSubtraction`, `ChamferUnion`,
//! `ChamferIntersection`, `ChamferSubtraction`, `StairsUnion`,
//! `StairsIntersection`, `StairsSubtraction`, `ColumnsUnion`,
//! `ColumnsIntersection`, `ColumnsSubtraction`, `XOR`, `Morph`, `Pipe`,
//! `Engrave`, `Groove`, `Tongue`, `ExpSmoothUnion`, `ExpSmoothIntersection`,
//! `ExpSmoothSubtraction`.
//!
//! Transforms (4): `Translate`, `Rotate`, `Scale`, `ScaleNonUniform`.
//!
//! Modifiers (16): `Twist`, `Bend`, `RepeatInfinite`, `RepeatFinite`,
//! `Round`, `Onion`, `Elongate`, `Noise`, `Mirror`, `Displacement`, `Shear`,
//! `Revolution`, `Extrude`, `SweepBezier`, `Taper`, `PolarRepeat`.
//!
//! Structural markers (2): `PopTransform`, `End`.
//!
//! Unsupported opcodes return [`RefitError::UnsupportedOpcode`] so callers
//! can fall back to the [`crate::incremental::ParamDependencyIndex::refit_bvh`]
//! wrapper (SdfNode-based full recompile). The remaining `~10` opcodes
//! (advanced modifiers `ProjectiveTransform` / `LatticeDeform` /
//! `SdfSkinning` / `IcosahedralSymmetry` / `IFS` / `HeightmapDisplacement`
//! / `SurfaceRoughness`) are silently rendered as tiny-sphere fallbacks by
//! the BVH compiler itself, so they never appear in the bytecode. The
//! transparent modifiers (`Animated`, `OctantMirror`, `WithMaterial`) emit
//! no instruction and likewise are absent.
//!
//! ## Author
//!
//! Moroya Sakamoto

use super::aabb::{primitives as aabb_prims, AabbPacked};
use super::eval_bvh::CompiledSdfBvh;
use super::opcode::OpCode;
use glam::{Quat, Vec3};
use std::collections::HashSet;

/// Errors returned by the bytecode refit walker.
#[derive(Debug, thiserror::Error)]
pub enum RefitError {
    /// An opcode encountered in the bytecode is not covered by the refit
    /// walker (P1 / P2 slice, ~24 opcodes). Fall back to
    /// `ParamDependencyIndex::refit_bvh(&mut bvh, &sdf_node)`.
    #[error("opcode {opcode:?} at instruction {instruction_index} is not supported by refit; fall back to SdfNode-based refit_bvh")]
    UnsupportedOpcode {
        /// Offending opcode.
        opcode: OpCode,
        /// Instruction index where it was encountered.
        instruction_index: usize,
    },

    /// The value stack underflowed while consuming children for a binary /
    /// unary operator. Indicates malformed bytecode.
    #[error(
        "value stack underflow at instruction {instruction_index} ({opcode:?}); malformed bytecode"
    )]
    ValueStackUnderflow {
        /// Instruction index where the pop failed.
        instruction_index: usize,
        /// Offending opcode.
        opcode: OpCode,
    },

    /// `PopTransform` was reached with no matching transform / modifier on
    /// the transform stack. Indicates malformed bytecode.
    #[error("transform stack underflow at PopTransform (instruction {instruction_index})")]
    TransformStackUnderflow {
        /// Instruction index of the PopTransform that failed.
        instruction_index: usize,
    },

    /// A `dirty` index passed to `refit_partial` exceeds `bvh.instructions.len()`.
    #[error("dirty index {index} out of bounds ({len} instructions)")]
    DirtyIndexOutOfBounds {
        /// Requested index.
        index: usize,
        /// Number of instructions in the BVH.
        len: usize,
    },

    /// `bvh.aabbs.len()` and `bvh.instructions.len()` disagree.
    #[error("bvh internal invariant: aabbs.len()={aabbs} != instructions.len()={instructions}")]
    LengthMismatch {
        /// AABB vector length.
        aabbs: usize,
        /// Instruction vector length.
        instructions: usize,
    },
}

/// Compile-time helper: build `parent_indices` for every instruction in a
/// prebuilt BVH by simulating the same tape walk. `parent_indices[i]` is
/// `Some(parent_index)` when instruction `i` is a child in a CSG binary op
/// (parent is the op instruction that consumes it) or an inner instruction
/// of a transform / modifier (parent is the transform instruction). Root
/// instruction gets `None`.
///
/// This is intentionally decoupled from `BvhCompiler` so that the same walk
/// can populate the field on already-compiled BVHs (e.g. loaded from disk)
/// without re-running the SdfNode compile path.
///
/// # Errors
///
/// Returns `RefitError::UnsupportedOpcode` if an opcode outside the P1 /
/// P2 coverage slice is encountered.
pub fn build_parent_indices(bvh: &CompiledSdfBvh) -> Result<Vec<Option<u32>>, RefitError> {
    let n = bvh.instructions.len();
    let mut parent = vec![None; n];
    let mut value_stack: Vec<usize> = Vec::new();
    // (transform instruction index)
    let mut transform_stack: Vec<usize> = Vec::new();

    for (i, inst) in bvh.instructions.iter().enumerate() {
        let opcode = inst.opcode;
        match opcode {
            // Leaves — primitives push themselves onto the value stack.
            OpCode::Sphere
            | OpCode::Box3d
            | OpCode::Cylinder
            | OpCode::Torus
            | OpCode::Plane
            | OpCode::Capsule
            | OpCode::Cone
            | OpCode::Ellipsoid
            | OpCode::RoundedCone
            | OpCode::Pyramid
            | OpCode::Octahedron
            | OpCode::HexPrism
            | OpCode::Link => {
                // If inside a transform, the top of transform_stack becomes
                // parent for this primitive.
                if let Some(&t) = transform_stack.last() {
                    parent[i] = Some(t as u32);
                }
                value_stack.push(i);
            }
            // CSG binary — pop 2 children, record parent as this instruction.
            op if is_supported_binary(op) => {
                let b = value_stack.pop().ok_or(RefitError::ValueStackUnderflow {
                    instruction_index: i,
                    opcode,
                })?;
                let a = value_stack.pop().ok_or(RefitError::ValueStackUnderflow {
                    instruction_index: i,
                    opcode,
                })?;
                parent[a] = Some(i as u32);
                parent[b] = Some(i as u32);
                if let Some(&t) = transform_stack.last() {
                    parent[i] = Some(t as u32);
                }
                value_stack.push(i);
            }
            // Transforms / modifiers — push onto transform_stack; the child
            // will follow in the bytecode.
            op if is_supported_transform_or_modifier(op) => {
                if let Some(&t) = transform_stack.last() {
                    parent[i] = Some(t as u32);
                }
                transform_stack.push(i);
            }
            OpCode::PopTransform => {
                let t = transform_stack
                    .pop()
                    .ok_or(RefitError::TransformStackUnderflow {
                        instruction_index: i,
                    })?;
                // Consume the child AABB the transform wraps.
                let child = value_stack.pop().ok_or(RefitError::ValueStackUnderflow {
                    instruction_index: i,
                    opcode,
                })?;
                parent[child] = Some(t as u32);
                // PopTransform itself does not participate in the value
                // stack — the transform instruction takes its place.
                value_stack.push(t);
            }
            OpCode::End => break,
            other => {
                return Err(RefitError::UnsupportedOpcode {
                    opcode: other,
                    instruction_index: i,
                });
            }
        }
    }
    Ok(parent)
}

/// Compile-time companion to [`build_parent_indices`] that records the
/// inclusive end index of each instruction's subtree. See
/// [`crate::compiled::CompiledSdfBvh::subtree_end`] for the semantics.
///
/// # Errors
///
/// Returns `RefitError::UnsupportedOpcode` if an opcode outside the P1 /
/// P3 coverage slice is encountered, or `TransformStackUnderflow` when a
/// `PopTransform` appears with no matching transform.
pub fn build_subtree_ends(bvh: &CompiledSdfBvh) -> Result<Vec<u32>, RefitError> {
    let n = bvh.instructions.len();
    let mut ends: Vec<u32> = (0..n as u32).collect(); // default: end == self
    let mut transform_stack: Vec<usize> = Vec::new();

    for (i, inst) in bvh.instructions.iter().enumerate() {
        let opcode = inst.opcode;
        match opcode {
            op if is_supported_transform_or_modifier(op) => {
                transform_stack.push(i);
            }
            OpCode::PopTransform => {
                let t = transform_stack
                    .pop()
                    .ok_or(RefitError::TransformStackUnderflow {
                        instruction_index: i,
                    })?;
                // The transform's subtree ends at this PopTransform.
                ends[t] = i as u32;
            }
            OpCode::End => break,
            op if is_supported_binary(op) || op.is_primitive() => {
                // Nothing to do — end == self was the initial value.
            }
            other => {
                return Err(RefitError::UnsupportedOpcode {
                    opcode: other,
                    instruction_index: i,
                });
            }
        }
    }
    Ok(ends)
}

/// Recompute every AABB in `bvh` from `Instruction.params[]`.
///
/// Walks the bytecode linearly using a value stack + transform stack that
/// mirrors the original `BvhCompiler::compile_node` semantics but reads
/// primitive dimensions from the compiled instructions.
///
/// # Errors
///
/// - `RefitError::LengthMismatch` if the caller passed a BVH whose
///   `aabbs` / `instructions` lengths disagree.
/// - `RefitError::UnsupportedOpcode` for opcodes outside the P1 slice.
/// - `RefitError::ValueStackUnderflow` / `TransformStackUnderflow` if the
///   bytecode is malformed.
pub fn refit_all(bvh: &mut CompiledSdfBvh) -> Result<usize, RefitError> {
    if bvh.aabbs.len() != bvh.instructions.len() {
        return Err(RefitError::LengthMismatch {
            aabbs: bvh.aabbs.len(),
            instructions: bvh.instructions.len(),
        });
    }
    let dirty = None;
    walk_and_recompute(bvh, dirty.as_ref())
}

/// Recompute AABBs only for `dirty` instructions and their ancestor chain.
///
/// The tape walk itself is O(N) — every instruction is visited to keep the
/// value stack coherent — but only the instructions in the effective
/// "recompute set" (dirty ∪ their ancestors) have their AABBs recomputed.
/// Non-affected subtrees keep their cached AABBs, which are still pushed
/// onto the value stack so parent operators see consistent inputs.
///
/// True O(dirty × depth) refit (skipping non-affected subtree walks)
/// requires storing subtree extents and is future work (P3).
///
/// # Errors
///
/// Same as [`refit_all`], plus:
///
/// - `RefitError::DirtyIndexOutOfBounds` if any `dirty` index is beyond
///   `bvh.instructions.len()`.
pub fn refit_partial(bvh: &mut CompiledSdfBvh, dirty: &[usize]) -> Result<usize, RefitError> {
    if bvh.aabbs.len() != bvh.instructions.len() {
        return Err(RefitError::LengthMismatch {
            aabbs: bvh.aabbs.len(),
            instructions: bvh.instructions.len(),
        });
    }
    let n = bvh.instructions.len();
    for &d in dirty {
        if d >= n {
            return Err(RefitError::DirtyIndexOutOfBounds { index: d, len: n });
        }
    }

    // Fold ancestors via parent_indices — this is the "affected set".
    let mut affected: HashSet<usize> = HashSet::with_capacity(dirty.len() * 2);
    for &d in dirty {
        let mut cursor = Some(d);
        while let Some(idx) = cursor {
            if !affected.insert(idx) {
                break;
            }
            cursor = bvh
                .parent_indices
                .get(idx)
                .and_then(|p| p.map(|q| q as usize));
        }
    }

    walk_and_recompute(bvh, Some(&affected))
}

/// Core walker used by both `refit_all` (`affected = None`) and
/// `refit_partial` (`affected = Some(set)`). Returns the number of AABBs
/// actually recomputed.
fn walk_and_recompute(
    bvh: &mut CompiledSdfBvh,
    affected: Option<&HashSet<usize>>,
) -> Result<usize, RefitError> {
    let n = bvh.instructions.len();
    let mut value_stack: Vec<AabbPacked> = Vec::with_capacity(64);
    // (transform instruction index, opcode) — the AABB is written on
    // PopTransform.
    let mut transform_stack: Vec<usize> = Vec::with_capacity(32);
    let mut recomputed = 0usize;
    let mut scene_aabb: Option<AabbPacked> = None;

    let should_recompute = |i: usize| affected.map_or(true, |set| set.contains(&i));
    // Fast subtree skip: if we're about to enter a transform whose subtree
    // contains no dirty instruction (i.e. the transform itself is not in
    // the affected set — since affected includes all ancestors of dirty,
    // this implies no descendant is dirty either), jump past the whole
    // subtree and push the cached AABB. Requires `bvh.subtree_end` to be
    // populated; falls back to per-instruction walk otherwise.
    let has_subtree_ends =
        bvh.subtree_end.len() == n && affected.is_some() && !bvh.subtree_end.is_empty();

    let mut i = 0usize;
    while i < n {
        let inst = bvh.instructions[i];
        let opcode = inst.opcode;
        // Try to skip a transform / modifier subtree wholesale.
        if has_subtree_ends && is_supported_transform_or_modifier(opcode) && !should_recompute(i) {
            let end = bvh.subtree_end[i] as usize;
            if end > i {
                let aabb = bvh.aabbs[i];
                value_stack.push(aabb);
                scene_aabb = Some(match scene_aabb {
                    None => aabb,
                    Some(prev) => prev.union(&aabb),
                });
                i = end + 1;
                continue;
            }
        }
        match opcode {
            OpCode::Sphere
            | OpCode::Box3d
            | OpCode::Cylinder
            | OpCode::Torus
            | OpCode::Plane
            | OpCode::Capsule
            | OpCode::Cone
            | OpCode::Ellipsoid
            | OpCode::RoundedCone
            | OpCode::Pyramid
            | OpCode::Octahedron
            | OpCode::HexPrism
            | OpCode::Link => {
                let aabb = if should_recompute(i) {
                    let new_aabb = primitive_aabb(opcode, &inst.params);
                    bvh.aabbs[i] = new_aabb;
                    recomputed += 1;
                    new_aabb
                } else {
                    bvh.aabbs[i]
                };
                value_stack.push(aabb);
                scene_aabb = Some(match scene_aabb {
                    None => aabb,
                    Some(prev) => prev.union(&aabb),
                });
            }
            op if is_supported_binary(op) => {
                let b = value_stack.pop().ok_or(RefitError::ValueStackUnderflow {
                    instruction_index: i,
                    opcode,
                })?;
                let a = value_stack.pop().ok_or(RefitError::ValueStackUnderflow {
                    instruction_index: i,
                    opcode,
                })?;
                let aabb = if should_recompute(i) {
                    let new_aabb = csg_binary_aabb(opcode, &inst.params, a, b);
                    bvh.aabbs[i] = new_aabb;
                    recomputed += 1;
                    new_aabb
                } else {
                    bvh.aabbs[i]
                };
                value_stack.push(aabb);
                scene_aabb = Some(match scene_aabb {
                    None => aabb,
                    Some(prev) => prev.union(&aabb),
                });
            }
            op if is_supported_transform_or_modifier(op) => {
                transform_stack.push(i);
            }
            OpCode::PopTransform => {
                let t = transform_stack
                    .pop()
                    .ok_or(RefitError::TransformStackUnderflow {
                        instruction_index: i,
                    })?;
                let child = value_stack.pop().ok_or(RefitError::ValueStackUnderflow {
                    instruction_index: i,
                    opcode,
                })?;
                let t_inst = bvh.instructions[t];
                let aabb = if should_recompute(t) {
                    let new_aabb = transform_or_modifier_aabb(t_inst.opcode, &t_inst.params, child);
                    bvh.aabbs[t] = new_aabb;
                    recomputed += 1;
                    new_aabb
                } else {
                    bvh.aabbs[t]
                };
                value_stack.push(aabb);
                scene_aabb = Some(match scene_aabb {
                    None => aabb,
                    Some(prev) => prev.union(&aabb),
                });
            }
            OpCode::End => break,
            other => {
                return Err(RefitError::UnsupportedOpcode {
                    opcode: other,
                    instruction_index: i,
                });
            }
        }
        i += 1;
    }

    if let Some(aabb) = scene_aabb {
        bvh.scene_aabb = aabb;
    }
    Ok(recomputed)
}

/// Whether `opcode` is one of the binary CSG operators handled by
/// [`csg_binary_aabb`]. Excludes any unsupported binary ops.
fn is_supported_binary(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::Union
            | OpCode::Intersection
            | OpCode::Subtraction
            | OpCode::SmoothUnion
            | OpCode::SmoothIntersection
            | OpCode::SmoothSubtraction
            | OpCode::ChamferUnion
            | OpCode::ChamferIntersection
            | OpCode::ChamferSubtraction
            | OpCode::StairsUnion
            | OpCode::StairsIntersection
            | OpCode::StairsSubtraction
            | OpCode::ColumnsUnion
            | OpCode::ColumnsIntersection
            | OpCode::ColumnsSubtraction
            | OpCode::XOR
            | OpCode::Morph
            | OpCode::Pipe
            | OpCode::Engrave
            | OpCode::Groove
            | OpCode::Tongue
            | OpCode::ExpSmoothUnion
            | OpCode::ExpSmoothIntersection
            | OpCode::ExpSmoothSubtraction
    )
}

/// Whether `opcode` is a transform / modifier that the refit walker knows
/// how to propagate through `PopTransform`. Excludes transparent modifiers
/// (`Animated`, `OctantMirror`, `WithMaterial`) — the BVH compiler emits
/// no instruction for those so they never appear in bytecode.
fn is_supported_transform_or_modifier(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::Translate
            | OpCode::Rotate
            | OpCode::Scale
            | OpCode::ScaleNonUniform
            | OpCode::Twist
            | OpCode::Bend
            | OpCode::RepeatInfinite
            | OpCode::RepeatFinite
            | OpCode::Round
            | OpCode::Onion
            | OpCode::Elongate
            | OpCode::Noise
            | OpCode::Mirror
            | OpCode::Displacement
            | OpCode::Shear
            | OpCode::Revolution
            | OpCode::Extrude
            | OpCode::SweepBezier
            | OpCode::Taper
            | OpCode::PolarRepeat
    )
}

fn primitive_aabb(opcode: OpCode, params: &[f32; 7]) -> AabbPacked {
    match opcode {
        OpCode::Sphere => aabb_prims::sphere_aabb(params[0]),
        OpCode::Box3d => aabb_prims::box_aabb(Vec3::new(params[0], params[1], params[2])),
        OpCode::Cylinder => aabb_prims::cylinder_aabb(params[0], params[1]),
        OpCode::Torus => aabb_prims::torus_aabb(params[0], params[1]),
        OpCode::Plane => aabb_prims::plane_aabb(),
        OpCode::Capsule => {
            let a = Vec3::new(params[0], params[1], params[2]);
            let b = Vec3::new(params[3], params[4], params[5]);
            aabb_prims::capsule_aabb(a, b, params[6])
        }
        OpCode::Cone => {
            let r = params[0];
            let hh = params[1];
            AabbPacked::new(Vec3::new(-r, -hh, -r), Vec3::new(r, hh, r))
        }
        OpCode::Ellipsoid => {
            let radii = Vec3::new(params[0], params[1], params[2]);
            AabbPacked::new(-radii, radii)
        }
        OpCode::RoundedCone => {
            let max_r = params[0].max(params[1]);
            let hh = params[2];
            AabbPacked::new(Vec3::new(-max_r, -hh, -max_r), Vec3::new(max_r, hh, max_r))
        }
        OpCode::Pyramid => {
            let hh = params[0];
            AabbPacked::new(Vec3::new(-0.5, -hh, -0.5), Vec3::new(0.5, hh, 0.5))
        }
        OpCode::Octahedron => {
            let s = params[0];
            AabbPacked::new(Vec3::new(-s, -s, -s), Vec3::new(s, s, s))
        }
        OpCode::HexPrism => {
            let hex_r = params[0];
            let hh = params[1];
            AabbPacked::new(Vec3::new(-hex_r, -hex_r, -hh), Vec3::new(hex_r, hex_r, hh))
        }
        OpCode::Link => {
            let half_length = params[0];
            let r1 = params[1];
            let r2 = params[2];
            let extent = r1 + r2;
            AabbPacked::new(
                Vec3::new(-extent, -(half_length + extent), -r2),
                Vec3::new(extent, half_length + extent, r2),
            )
        }
        _ => AabbPacked::empty(),
    }
}

fn csg_binary_aabb(opcode: OpCode, params: &[f32; 7], a: AabbPacked, b: AabbPacked) -> AabbPacked {
    match opcode {
        // Union-shape ops that don't expand the union bound.
        OpCode::Union | OpCode::XOR | OpCode::Morph | OpCode::Pipe => a.union(&b),
        // Smooth blends expand the union AABB by their radius / k.
        OpCode::SmoothUnion => a.union(&b).expand(params[0]),
        OpCode::ChamferUnion => a.union(&b).expand(params[0]),
        OpCode::StairsUnion => a.union(&b).expand(params[0]),
        OpCode::ColumnsUnion => a.union(&b).expand(params[0]),
        OpCode::ExpSmoothUnion => a.union(&b).expand(params[0]),
        // Intersection-shape ops shrink to the geometric intersection.
        OpCode::Intersection
        | OpCode::SmoothIntersection
        | OpCode::ChamferIntersection
        | OpCode::StairsIntersection
        | OpCode::ColumnsIntersection
        | OpCode::ExpSmoothIntersection => a.intersection(&b),
        // Subtraction / carving ops keep the LHS bound (RHS only removes).
        OpCode::Subtraction
        | OpCode::SmoothSubtraction
        | OpCode::ChamferSubtraction
        | OpCode::StairsSubtraction
        | OpCode::ColumnsSubtraction
        | OpCode::ExpSmoothSubtraction
        | OpCode::Engrave
        | OpCode::Groove
        | OpCode::Tongue => a,
        _ => a.union(&b),
    }
}

fn transform_or_modifier_aabb(opcode: OpCode, params: &[f32; 7], child: AabbPacked) -> AabbPacked {
    match opcode {
        // === Transforms ===
        OpCode::Translate => {
            let offset = Vec3::new(params[0], params[1], params[2]);
            child.translate(offset)
        }
        OpCode::Rotate => {
            let quat = Quat::from_xyzw(params[0], params[1], params[2], params[3]);
            child.rotate(quat)
        }
        OpCode::Scale => {
            // params[0] = 1/factor, params[1] = factor
            child.scale(params[1])
        }
        OpCode::ScaleNonUniform => {
            // params[0..3] = 1/sx, 1/sy, 1/sz — invert to recover factors.
            let sx = 1.0 / params[0];
            let sy = 1.0 / params[1];
            let sz = 1.0 / params[2];
            child.scale_nonuniform(Vec3::new(sx, sy, sz))
        }
        // === Surface-offset modifiers (radial expand) ===
        OpCode::Round => child.expand(params[0]),
        OpCode::Onion => child.expand(params[0]),
        OpCode::Displacement => child.expand(params[0].abs()),
        OpCode::Noise => child.expand(params[0]),
        // === Point-modifying modifiers with conservative expand ===
        OpCode::Twist => {
            let max_extent = child.half_size().max_element();
            child.expand(max_extent * params[0].abs() * 0.5)
        }
        OpCode::Bend => {
            let max_extent = child.half_size().max_element();
            child.expand(max_extent * params[0].abs())
        }
        OpCode::Taper => {
            let max_extent = child.half_size().max_element();
            child.expand(max_extent * params[0].abs())
        }
        OpCode::Shear => {
            let max_shear = params[0].abs().max(params[1].abs()).max(params[2].abs());
            let half = child.half_size();
            let expand = half.length() * max_shear;
            child.expand(expand)
        }
        OpCode::Elongate => {
            let amount = Vec3::new(params[0], params[1], params[2]);
            AabbPacked::new(child.min() - amount, child.max() + amount)
        }
        // === Repeat / mirror / symmetry ===
        OpCode::RepeatInfinite => AabbPacked::infinite(),
        OpCode::RepeatFinite => {
            // params[0..3] = counts (as f32), params[3..6] = spacing
            let cx = params[0];
            let cy = params[1];
            let cz = params[2];
            let sx = params[3];
            let sy = params[4];
            let sz = params[5];
            let expand = Vec3::new(cx * sx, cy * sy, cz * sz);
            AabbPacked::new(child.min() - expand, child.max() + expand)
        }
        OpCode::Mirror => {
            // params[0..3] = axis mask (non-zero → mirrored). Match the BVH
            // compiler's Mirror { child, axes } branch.
            let ax = params[0];
            let ay = params[1];
            let az = params[2];
            let cmin = child.min();
            let cmax = child.max();
            let ex = cmax.x.abs().max(cmin.x.abs());
            let ey = cmax.y.abs().max(cmin.y.abs());
            let ez = cmax.z.abs().max(cmin.z.abs());
            AabbPacked::new(
                Vec3::new(
                    if ax != 0.0 { -ex } else { cmin.x },
                    if ay != 0.0 { -ey } else { cmin.y },
                    if az != 0.0 { -ez } else { cmin.z },
                ),
                Vec3::new(
                    if ax != 0.0 { ex } else { cmax.x },
                    if ay != 0.0 { ey } else { cmax.y },
                    if az != 0.0 { ez } else { cmax.z },
                ),
            )
        }
        OpCode::PolarRepeat => {
            let half = child.half_size();
            let center = child.center();
            let max_r = half.x.max(half.z) + center.x.abs().max(center.z.abs());
            AabbPacked::new(
                Vec3::new(-max_r, child.min().y, -max_r),
                Vec3::new(max_r, child.max().y, max_r),
            )
        }
        // === 2D → 3D lifts ===
        OpCode::Revolution => {
            let offset = params[0];
            let max_r = child.max().x.abs().max(child.min().x.abs()) + offset.abs();
            AabbPacked::new(
                Vec3::new(-max_r, child.min().y, -max_r),
                Vec3::new(max_r, child.max().y, max_r),
            )
        }
        OpCode::Extrude => {
            let hh = params[0];
            AabbPacked::new(
                Vec3::new(child.min().x, child.min().y, -hh),
                Vec3::new(child.max().x, child.max().y, hh),
            )
        }
        OpCode::SweepBezier => {
            // params[0..6] = p0.x, p0.z, p1.x, p1.z, p2.x, p2.z
            let bmin_x = params[0].min(params[2]).min(params[4]);
            let bmax_x = params[0].max(params[2]).max(params[4]);
            let bmin_z = params[1].min(params[3]).min(params[5]);
            let bmax_z = params[1].max(params[3]).max(params[5]);
            let max_perp = child.max().x.abs().max(child.min().x.abs());
            AabbPacked::new(
                Vec3::new(bmin_x - max_perp, child.min().y, bmin_z - max_perp),
                Vec3::new(bmax_x + max_perp, child.max().y, bmax_z + max_perp),
            )
        }
        _ => child,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled::CompiledSdfBvh;
    use crate::types::SdfNode;

    fn build_two_sphere_union() -> (SdfNode, CompiledSdfBvh) {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(0.5).translate(3.0, 0.0, 0.0);
        let scene = a.union(b);
        let bvh = CompiledSdfBvh::compile(&scene);
        (scene, bvh)
    }

    #[test]
    fn refit_all_produces_same_aabbs_as_fresh_compile() {
        let (scene, mut bvh) = build_two_sphere_union();
        let baseline = CompiledSdfBvh::compile(&scene);
        let count = refit_all(&mut bvh).expect("refit_all must succeed");
        assert!(count > 0);
        for (a, b) in bvh.aabbs.iter().zip(baseline.aabbs.iter()) {
            assert!((a.min() - b.min()).length() < 1e-4);
            assert!((a.max() - b.max()).length() < 1e-4);
        }
        assert!((bvh.scene_aabb.min() - baseline.scene_aabb.min()).length() < 1e-4);
    }

    #[test]
    fn refit_all_reflects_param_mutation() {
        let (_, mut bvh) = build_two_sphere_union();
        // Find first Sphere and enlarge its radius directly in the bytecode.
        let sphere_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Sphere)
            .unwrap();
        bvh.instructions[sphere_idx].params[0] = 4.0;
        refit_all(&mut bvh).unwrap();
        let aabb = bvh.aabbs[sphere_idx];
        assert!((aabb.max() - Vec3::splat(4.0)).length() < 1e-4);
        assert!((aabb.min() - Vec3::splat(-4.0)).length() < 1e-4);
    }

    #[test]
    fn refit_all_rejects_unsupported_opcode() {
        // ProjectiveTransform never appears in real BVH bytecode (the BVH
        // compiler falls it back to a tiny sphere), but we can inject the
        // opcode manually to exercise the UnsupportedOpcode path.
        let (_, mut bvh) = build_two_sphere_union();
        bvh.instructions[0].opcode = OpCode::ProjectiveTransform;
        let err = refit_all(&mut bvh)
            .expect_err("ProjectiveTransform must fail refit (outside P1/P3 slice)");
        match err {
            RefitError::UnsupportedOpcode { opcode, .. } => {
                assert_eq!(opcode, OpCode::ProjectiveTransform);
            }
            other => panic!("expected UnsupportedOpcode, got {other:?}"),
        }
    }

    #[test]
    fn refit_partial_recomputes_only_ancestor_chain() {
        let (_, mut bvh) = build_two_sphere_union();
        let baseline_count = refit_all(&mut bvh).unwrap();
        assert!(baseline_count > 0);

        let sphere_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Sphere)
            .unwrap();
        // Refit just this primitive's chain.
        let partial_count = refit_partial(&mut bvh, &[sphere_idx]).unwrap();
        assert!(partial_count > 0);
        assert!(
            partial_count < baseline_count,
            "partial refit should recompute fewer instructions than full ({partial_count} vs {baseline_count})"
        );
    }

    #[test]
    fn refit_partial_matches_refit_all_when_dirty_is_everything() {
        let (scene, mut bvh_partial) = build_two_sphere_union();
        let mut bvh_full = CompiledSdfBvh::compile(&scene);
        let all_indices: Vec<usize> = (0..bvh_partial.instructions.len()).collect();
        refit_partial(&mut bvh_partial, &all_indices).unwrap();
        refit_all(&mut bvh_full).unwrap();
        for (a, b) in bvh_partial.aabbs.iter().zip(bvh_full.aabbs.iter()) {
            assert!((a.min() - b.min()).length() < 1e-4);
            assert!((a.max() - b.max()).length() < 1e-4);
        }
    }

    #[test]
    fn refit_partial_rejects_out_of_bounds_dirty() {
        let (_, mut bvh) = build_two_sphere_union();
        let err = refit_partial(&mut bvh, &[9999]).expect_err("out of bounds must fail");
        assert!(matches!(err, RefitError::DirtyIndexOutOfBounds { .. }));
    }

    #[test]
    fn build_parent_indices_root_has_no_parent() {
        let (_, bvh) = build_two_sphere_union();
        let parents = build_parent_indices(&bvh).unwrap();
        // The Union instruction is the outermost — its parent is None.
        let union_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Union)
            .unwrap();
        assert_eq!(parents[union_idx], None);
    }

    #[test]
    fn build_parent_indices_child_points_to_parent_op() {
        let scene = SdfNode::sphere(1.0).union(SdfNode::sphere(0.5));
        let bvh = CompiledSdfBvh::compile(&scene);
        let parents = build_parent_indices(&bvh).unwrap();
        let union_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Union)
            .unwrap();
        let first_sphere = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Sphere)
            .unwrap();
        assert_eq!(parents[first_sphere], Some(union_idx as u32));
    }

    // ── P3 opcode coverage smoke tests ──

    fn compile_and_refit(scene: SdfNode) -> Result<CompiledSdfBvh, RefitError> {
        let mut bvh = CompiledSdfBvh::compile(&scene);
        refit_all(&mut bvh)?;
        Ok(bvh)
    }

    #[test]
    fn refit_supports_twist_modifier() {
        let scene = SdfNode::sphere(1.0).twist(0.5);
        let bvh = compile_and_refit(scene).expect("Twist refit must succeed");
        assert!(bvh.scene_aabb.is_valid());
    }

    #[test]
    fn refit_supports_bend_modifier() {
        let scene = SdfNode::box3d(1.0, 1.0, 1.0).bend(0.3);
        let bvh = compile_and_refit(scene).expect("Bend refit must succeed");
        assert!(bvh.scene_aabb.is_valid());
    }

    #[test]
    fn refit_supports_repeat_finite_modifier() {
        let scene = SdfNode::sphere(0.5).repeat_finite([2, 2, 2], Vec3::splat(1.5));
        let bvh = compile_and_refit(scene).expect("RepeatFinite refit must succeed");
        assert!(bvh.scene_aabb.is_valid());
        assert!(bvh.scene_aabb.half_size().max_element() > 0.5);
    }

    #[test]
    fn refit_supports_repeat_infinite_modifier() {
        let scene = SdfNode::sphere(0.5).repeat_infinite(2.0, 2.0, 2.0);
        let bvh = compile_and_refit(scene).expect("RepeatInfinite refit must succeed");
        assert_eq!(
            bvh.scene_aabb.min().x,
            f32::MIN,
            "RepeatInfinite must yield an infinite scene AABB"
        );
    }

    #[test]
    fn refit_supports_onion_and_displacement_modifiers() {
        let scene = SdfNode::sphere(1.0).onion(0.1);
        let bvh = compile_and_refit(scene).expect("Onion refit must succeed");
        assert!(bvh.scene_aabb.half_size().max_element() >= 1.1);

        let scene = SdfNode::sphere(1.0).displacement(0.2);
        let bvh = compile_and_refit(scene).expect("Displacement refit must succeed");
        assert!(bvh.scene_aabb.half_size().max_element() >= 1.0);
    }

    #[test]
    fn refit_supports_elongate_modifier() {
        let scene = SdfNode::sphere(0.5).elongate(1.0, 0.0, 0.0);
        let bvh = compile_and_refit(scene).expect("Elongate refit must succeed");
        assert!(bvh.scene_aabb.max().x >= 1.5);
    }

    #[test]
    fn refit_supports_mirror_modifier() {
        let scene = SdfNode::sphere(0.5)
            .translate(2.0, 0.0, 0.0)
            .mirror(true, false, false);
        let bvh = compile_and_refit(scene).expect("Mirror refit must succeed");
        assert!(bvh.scene_aabb.is_valid());
    }

    #[test]
    fn refit_supports_extrude_and_revolution() {
        // Extrude / Revolution are validated smoke tests — walker must
        // terminate and produce a valid AABB. Exact bounds depend on how the
        // BVH compiler applies these opcodes to a 3D child (implementation
        // treats the child bounds as-is and rewraps in Z / X respectively).
        let scene = SdfNode::sphere(0.5).extrude(1.0);
        let bvh = compile_and_refit(scene).expect("Extrude refit must succeed");
        assert!(bvh.scene_aabb.is_valid());

        let scene = SdfNode::sphere(0.5).revolution(2.0);
        let bvh = compile_and_refit(scene).expect("Revolution refit must succeed");
        assert!(bvh.scene_aabb.max().x >= 2.0);
    }

    #[test]
    fn refit_supports_polar_repeat_modifier() {
        let scene = SdfNode::sphere(0.5)
            .translate(1.5, 0.0, 0.0)
            .polar_repeat(6);
        let bvh = compile_and_refit(scene).expect("PolarRepeat refit must succeed");
        assert!(bvh.scene_aabb.is_valid());
    }

    #[test]
    fn refit_supports_chamfer_union() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(0.5).translate(1.5, 0.0, 0.0);
        let scene = a.chamfer_union(b, 0.2);
        let bvh = compile_and_refit(scene).expect("ChamferUnion refit must succeed");
        assert!(bvh.scene_aabb.is_valid());
    }

    #[test]
    fn refit_supports_stairs_union() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(0.5).translate(1.5, 0.0, 0.0);
        let scene = a.stairs_union(b, 0.15, 3.0);
        let bvh = compile_and_refit(scene).expect("StairsUnion refit must succeed");
        assert!(bvh.scene_aabb.is_valid());
    }

    #[test]
    fn refit_supports_columns_pipe_engrave_variants() {
        // Sanity: three lesser-used binary variants should each refit.
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(0.5).translate(1.5, 0.0, 0.0);
        let scene = a.clone().columns_union(b.clone(), 0.2, 3.0);
        compile_and_refit(scene).expect("ColumnsUnion refit must succeed");

        let scene = a.clone().pipe(b.clone(), 0.1);
        compile_and_refit(scene).expect("Pipe refit must succeed");

        let scene = a.engrave(b, 0.15);
        compile_and_refit(scene).expect("Engrave refit must succeed");
    }

    #[test]
    fn refit_supports_exp_smooth_union_intersection_subtraction() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(0.5).translate(1.5, 0.0, 0.0);
        compile_and_refit(a.clone().exp_smooth_union(b.clone(), 0.2))
            .expect("ExpSmoothUnion refit must succeed");
        compile_and_refit(a.clone().exp_smooth_intersection(b.clone(), 0.2))
            .expect("ExpSmoothIntersection refit must succeed");
        compile_and_refit(a.exp_smooth_subtract(b, 0.2))
            .expect("ExpSmoothSubtraction refit must succeed");
    }

    #[test]
    fn refit_partial_walks_ancestor_chain_with_modifiers() {
        // Ensure partial refit still terminates correctly when the tree
        // contains a modifier chain (previously P1-only path panicked here).
        let scene = SdfNode::sphere(1.0).twist(0.5).bend(0.3);
        let mut bvh = CompiledSdfBvh::compile(&scene);
        let sphere_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Sphere)
            .unwrap();
        let count = refit_partial(&mut bvh, &[sphere_idx]).unwrap();
        // Expect: Sphere + Twist ancestor + Bend ancestor = 3 recomputes.
        assert_eq!(count, 3);
    }

    // ── P4 subtree skip tests ──

    #[test]
    fn subtree_end_populates_on_compile() {
        // Union(Sphere, Translate(...).Box) should populate a non-empty
        // subtree_end vector matching the instruction count.
        let (_, bvh) = build_two_sphere_union();
        assert_eq!(
            bvh.subtree_end.len(),
            bvh.instructions.len(),
            "subtree_end must cover every instruction"
        );
        // Translate's subtree_end should point at the matching PopTransform.
        let translate_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Translate);
        if let Some(idx) = translate_idx {
            let end = bvh.subtree_end[idx] as usize;
            assert!(end > idx, "transform subtree_end must be > its own index");
            assert_eq!(bvh.instructions[end].opcode, OpCode::PopTransform);
        }
    }

    #[test]
    fn refit_partial_skips_untouched_transform_subtree() {
        // Union(dirty_sphere, Translate.Box). Marking only the sphere dirty
        // should let the walker skip the Translate/Box/PopTransform triplet
        // wholesale — the recompute count stays low but the Translate branch
        // still needs its ancestor Union recomputed.
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::box3d(0.5, 0.5, 0.5).translate(3.0, 0.0, 0.0);
        let scene = a.union(b);
        let mut bvh = CompiledSdfBvh::compile(&scene);
        let sphere_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Sphere)
            .unwrap();
        // Enlarge just the sphere, then partial-refit.
        bvh.instructions[sphere_idx].params[0] = 2.0;
        let count = refit_partial(&mut bvh, &[sphere_idx]).unwrap();
        // Expect: Sphere + Union ancestor = 2 recomputes (Translate/Box/Pop
        // are skipped). No recomputes for the box subtree.
        assert_eq!(
            count, 2,
            "only dirty leaf + Union ancestor should recompute"
        );
        let sphere_aabb = bvh.aabbs[sphere_idx];
        assert!((sphere_aabb.max().x - 2.0).abs() < 1e-4);
    }

    #[test]
    fn refit_partial_still_produces_correct_scene_aabb() {
        // Compare partial-with-skip vs full refit: identical results.
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::box3d(0.5, 0.5, 0.5).translate(3.0, 0.0, 0.0);
        let c = SdfNode::sphere(0.7).translate(0.0, 4.0, 0.0);
        let scene = a.union(b).union(c);

        let mut bvh_partial = CompiledSdfBvh::compile(&scene);
        let mut bvh_full = CompiledSdfBvh::compile(&scene);
        // Mutate a single sphere in both to keep them in sync, then refit.
        let sphere_idx = bvh_partial
            .instructions
            .iter()
            .position(|i| i.opcode == OpCode::Sphere)
            .unwrap();
        bvh_partial.instructions[sphere_idx].params[0] = 1.5;
        bvh_full.instructions[sphere_idx].params[0] = 1.5;
        refit_partial(&mut bvh_partial, &[sphere_idx]).unwrap();
        refit_all(&mut bvh_full).unwrap();
        // Sphere AABB updated in partial version.
        assert!((bvh_partial.aabbs[sphere_idx].max().x - 1.5).abs() < 1e-4);
        // Scene AABB matches between partial (with skip) and full walk.
        assert!(
            (bvh_partial.scene_aabb.max() - bvh_full.scene_aabb.max()).length() < 1e-4,
            "scene AABB must agree"
        );
    }
}
