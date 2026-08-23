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
//! ## Opcode coverage (P1 initial slice, ~24)
//!
//! Primitives: `Sphere`, `Box3d`, `Cylinder`, `Torus`, `Plane`, `Capsule`,
//! `Cone`, `Ellipsoid`, `RoundedCone`, `Pyramid`, `Octahedron`, `HexPrism`,
//! `Link` (13).
//!
//! CSG binary: `Union`, `Intersection`, `Subtraction`, `SmoothUnion`,
//! `SmoothIntersection`, `SmoothSubtraction`, `XOR`, `Morph` (8).
//!
//! Transforms: `Translate`, `Rotate`, `Scale`, `ScaleNonUniform` (4).
//!
//! Modifiers: `Round` (1).
//!
//! Structural markers: `PopTransform`, `End` (2).
//!
//! Unsupported opcodes return [`RefitError::UnsupportedOpcode`] so callers
//! can fall back to the [`crate::incremental::ParamDependencyIndex::refit_bvh`]
//! wrapper (SdfNode-based full recompile) for scenes that use richer
//! modifiers or CSG variants. Extending coverage is future work (P3).
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
            OpCode::Union
            | OpCode::Intersection
            | OpCode::Subtraction
            | OpCode::SmoothUnion
            | OpCode::SmoothIntersection
            | OpCode::SmoothSubtraction
            | OpCode::XOR
            | OpCode::Morph => {
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
            OpCode::Translate
            | OpCode::Rotate
            | OpCode::Scale
            | OpCode::ScaleNonUniform
            | OpCode::Round => {
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

    for i in 0..n {
        let inst = bvh.instructions[i];
        let opcode = inst.opcode;
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
            OpCode::Union
            | OpCode::Intersection
            | OpCode::Subtraction
            | OpCode::SmoothUnion
            | OpCode::SmoothIntersection
            | OpCode::SmoothSubtraction
            | OpCode::XOR
            | OpCode::Morph => {
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
            OpCode::Translate
            | OpCode::Rotate
            | OpCode::Scale
            | OpCode::ScaleNonUniform
            | OpCode::Round => {
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
    }

    if let Some(aabb) = scene_aabb {
        bvh.scene_aabb = aabb;
    }
    Ok(recomputed)
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

fn csg_binary_aabb(opcode: OpCode, _params: &[f32; 7], a: AabbPacked, b: AabbPacked) -> AabbPacked {
    match opcode {
        OpCode::Union | OpCode::XOR | OpCode::Morph => a.union(&b),
        OpCode::Intersection | OpCode::SmoothIntersection => a.intersection(&b),
        OpCode::Subtraction | OpCode::SmoothSubtraction => a,
        OpCode::SmoothUnion => a.union(&b).expand(_params[0]),
        _ => a.union(&b),
    }
}

fn transform_or_modifier_aabb(opcode: OpCode, params: &[f32; 7], child: AabbPacked) -> AabbPacked {
    match opcode {
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
        OpCode::Round => child.expand(params[0]),
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
        // Twist is intentionally outside the P1 coverage — construct a
        // scene that includes it and ensure refit reports the error.
        let scene = SdfNode::sphere(1.0).twist(0.5);
        let mut bvh = CompiledSdfBvh::compile(&scene);
        let err = refit_all(&mut bvh).expect_err("Twist must fail P1 refit");
        match err {
            RefitError::UnsupportedOpcode { opcode, .. } => {
                assert_eq!(opcode, OpCode::Twist);
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
}
