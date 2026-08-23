//! Incremental parameter update support for `CompiledSdf`.
//!
//! This module provides a *binding layer* between [`ParamId`] (parametric
//! variable identifier managed by [`ConstraintSolver`]) and the raw `f32`
//! parameter slots inside [`crate::compiled::Instruction`]. It enables incremental workflows
//! such as animation and constraint-driven update, where a single parameter
//! change should only recompute the affected subtree instead of the whole
//! SDF tree.
//!
//! ## Scope
//!
//! Case A (foundation, 2026-08-23):
//!
//! - Manual binding registration (`bind`) — the caller declares which
//!   `Instruction.params[slot]` belongs to which `ParamId`.
//! - Value application (`apply` / `apply_all`) — write updated parameter
//!   values into the bytecode in place.
//! - Dirty tracking (`dirty_params` / `dirty_instructions`) — the caller can
//!   iterate affected instruction indices to hook cache invalidation.
//!
//! Case B (cache glue, 2026-08-23):
//!
//! - `affected_aabb` / `dirty_aabb` — union AABBs from `CompiledSdfBvh` for a
//!   single parameter or the current dirty set.
//! - `invalidate_chunked_cache` — glue that computes `dirty_aabb`, forwards
//!   it to `ChunkedMeshCache::invalidate_region`, and clears the dirty set.
//!
//! Case B.4 wrapper (SdfNode-based full refit, 2026-08-23):
//!
//! - `refit_bvh(&mut bvh, &sdf_node)` — replaces the BVH in place by
//!   re-running `CompiledSdfBvh::try_compile(sdf_node)`. **Important**: this
//!   reads AABB source values from the `SdfNode`, not from
//!   `Instruction.params[]`. If the caller mutated bytecode via `apply` /
//!   `apply_all` they must independently synchronise the `SdfNode` (there is
//!   no built-in `ParamId → SdfNode` field binding) before calling this
//!   wrapper, otherwise the refit produces AABBs that reflect the *original*
//!   SdfNode values.
//!
//! Deferred: true partial refit (bytecode-driven, O(dirty × depth) with
//! per-opcode AABB compute reused from the BVH compiler) and subtree eval
//! auto-skip (not sound per-point).
//!
//! ## Author
//!
//! Moroya Sakamoto

use crate::cache::ChunkedMeshCache;
use crate::compiled::{AabbPacked, CompileError, CompiledSdf, CompiledSdfBvh};
use crate::constraint::{ConstraintSolver, ParamId};
use crate::types::SdfNode;
use std::collections::{HashMap, HashSet};

/// A single binding site inside `CompiledSdf.instructions[i].params[slot]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionSlot {
    /// Index into `CompiledSdf::instructions`.
    pub instruction_index: usize,
    /// Slot within `Instruction::params` (0..7).
    pub param_slot: u8,
}

impl InstructionSlot {
    /// Create a new binding site.
    #[must_use]
    pub const fn new(instruction_index: usize, param_slot: u8) -> Self {
        Self {
            instruction_index,
            param_slot,
        }
    }
}

/// Errors returned by [`ParamDependencyIndex`] mutation methods.
#[derive(Debug, thiserror::Error)]
pub enum IncrementalError {
    /// Instruction index is beyond `CompiledSdf::instructions.len()`.
    #[error("instruction index {index} out of bounds ({len} instructions)")]
    InstructionOutOfBounds {
        /// Requested index.
        index: usize,
        /// Number of instructions in the compiled SDF.
        len: usize,
    },

    /// Parameter slot is outside the valid `0..7` range.
    #[error("param slot {slot} out of range (valid: 0..7)")]
    ParamSlotOutOfRange {
        /// Requested slot.
        slot: u8,
    },

    /// Parameter id is beyond the solver's parameter vector length.
    #[error("param id {pid:?} out of range ({len} params in solver)")]
    ParamIdOutOfRange {
        /// Requested id.
        pid: ParamId,
        /// Parameter count in the solver.
        len: usize,
    },
}

/// Reverse index from [`ParamId`] to the instruction slots that consume it.
///
/// Bindings are registered manually by the caller (e.g. the DSL front-end or
/// a scene loader) — the bytecode itself does not carry `ParamId` references.
/// After binding, `apply` / `apply_all` write updated values into the
/// bytecode in place, and `dirty_*` iterators expose which instructions were
/// touched so the caller can invalidate downstream caches.
#[derive(Debug, Default, Clone)]
pub struct ParamDependencyIndex {
    bindings: HashMap<ParamId, Vec<InstructionSlot>>,
    dirty: HashSet<ParamId>,
}

impl ParamDependencyIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a binding between `pid` and `Instruction.params[slot]` at
    /// `instruction_index`.
    ///
    /// # Errors
    ///
    /// Returns `IncrementalError::InstructionOutOfBounds` if `instruction_index`
    /// is beyond `compiled.instructions.len()`, or `ParamSlotOutOfRange` if
    /// `param_slot >= 7`.
    pub fn bind(
        &mut self,
        pid: ParamId,
        compiled: &CompiledSdf,
        instruction_index: usize,
        param_slot: u8,
    ) -> Result<(), IncrementalError> {
        if instruction_index >= compiled.instructions.len() {
            return Err(IncrementalError::InstructionOutOfBounds {
                index: instruction_index,
                len: compiled.instructions.len(),
            });
        }
        if param_slot >= 7 {
            return Err(IncrementalError::ParamSlotOutOfRange { slot: param_slot });
        }
        self.bindings
            .entry(pid)
            .or_default()
            .push(InstructionSlot::new(instruction_index, param_slot));
        Ok(())
    }

    /// Return the binding sites for `pid`, or an empty slice if unbound.
    #[must_use]
    pub fn bindings_of(&self, pid: ParamId) -> &[InstructionSlot] {
        self.bindings.get(&pid).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Number of distinct `ParamId` values with at least one binding.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.bindings.len()
    }

    /// Total number of registered `(ParamId, instruction, slot)` triples.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.bindings.values().map(Vec::len).sum()
    }

    /// Write `value` into every instruction slot bound to `pid`, and mark
    /// `pid` as dirty for cache invalidation.
    ///
    /// Unbound `pid` is a silent no-op (does not mark dirty).
    ///
    /// # Errors
    ///
    /// Returns `InstructionOutOfBounds` if a stored binding is beyond
    /// `compiled.instructions.len()` (should not happen unless the compiled
    /// program was replaced with a shorter one).
    pub fn apply(
        &mut self,
        compiled: &mut CompiledSdf,
        pid: ParamId,
        value: f32,
    ) -> Result<(), IncrementalError> {
        let Some(slots) = self.bindings.get(&pid) else {
            return Ok(());
        };
        let len = compiled.instructions.len();
        for slot in slots {
            if slot.instruction_index >= len {
                return Err(IncrementalError::InstructionOutOfBounds {
                    index: slot.instruction_index,
                    len,
                });
            }
            compiled.instructions[slot.instruction_index].params[slot.param_slot as usize] = value;
        }
        self.dirty.insert(pid);
        Ok(())
    }

    /// Push every parameter value from `solver` into the bytecode via the
    /// registered bindings. Marks every bound `pid` as dirty.
    ///
    /// # Errors
    ///
    /// Returns `ParamIdOutOfRange` if a bound `pid` is not addressable by the
    /// solver, or `InstructionOutOfBounds` (see [`apply`](Self::apply)).
    pub fn apply_all(
        &mut self,
        compiled: &mut CompiledSdf,
        solver: &ConstraintSolver,
    ) -> Result<(), IncrementalError> {
        let solver_len = solver.param_count();
        let bound_pids: Vec<ParamId> = self.bindings.keys().copied().collect();
        for pid in bound_pids {
            if pid.as_index() >= solver_len {
                return Err(IncrementalError::ParamIdOutOfRange {
                    pid,
                    len: solver_len,
                });
            }
            #[allow(clippy::cast_possible_truncation)]
            let value = solver.get(pid) as f32;
            self.apply(compiled, pid, value)?;
        }
        Ok(())
    }

    /// Iterate over parameters modified since the last [`mark_clean`](Self::mark_clean).
    pub fn dirty_params(&self) -> impl Iterator<Item = ParamId> + '_ {
        self.dirty.iter().copied()
    }

    /// Iterate over instruction indices whose slot was written since the last
    /// [`mark_clean`](Self::mark_clean). Order is unspecified; duplicates are
    /// possible when a single instruction has multiple slots bound to
    /// different dirty `ParamId`s.
    pub fn dirty_instructions(&self) -> impl Iterator<Item = usize> + '_ {
        self.dirty
            .iter()
            .filter_map(|pid| self.bindings.get(pid))
            .flat_map(|slots| slots.iter().map(|s| s.instruction_index))
    }

    /// Clear the dirty set. Call after downstream caches have been invalidated.
    pub fn mark_clean(&mut self) {
        self.dirty.clear();
    }

    /// Whether at least one parameter has been modified since the last
    /// [`mark_clean`](Self::mark_clean).
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Return the union of per-instruction AABBs covering every binding of `pid`.
    ///
    /// The returned AABB is the union of `bvh.aabbs[i]` for each bound
    /// instruction. For primitives nested inside transforms, `bvh.aabbs[i]`
    /// stores the **local** primitive AABB (before the parent transform), so
    /// this method understates the world-space region a downstream cache
    /// needs to invalidate. Callers that require world-space bounds for a
    /// transformed subtree should either bind the enclosing transform
    /// instruction as well or fall back to `bvh.scene_aabb`. Removing this
    /// caveat is tracked as case B.4 (partial BVH refit).
    ///
    /// Returns `None` if `pid` is unbound, has zero valid instruction indices,
    /// or every referenced AABB is empty. Instruction indices beyond
    /// `bvh.aabbs.len()` are silently skipped so stale bindings do not panic.
    #[must_use]
    pub fn affected_aabb(&self, pid: ParamId, bvh: &CompiledSdfBvh) -> Option<AabbPacked> {
        let slots = self.bindings.get(&pid)?;
        union_of_instruction_aabbs(slots.iter().map(|s| s.instruction_index), bvh)
    }

    /// Return the union of per-instruction AABBs covering every currently
    /// dirty `ParamId`.
    ///
    /// Returns `None` when no dirty parameter maps to a valid, non-empty AABB.
    #[must_use]
    pub fn dirty_aabb(&self, bvh: &CompiledSdfBvh) -> Option<AabbPacked> {
        let indices = self
            .dirty
            .iter()
            .filter_map(|pid| self.bindings.get(pid))
            .flat_map(|slots| slots.iter().map(|s| s.instruction_index));
        union_of_instruction_aabbs(indices, bvh)
    }

    /// Mark every `ChunkedMeshCache` chunk overlapping the current dirty AABB
    /// as dirty, then clear the dirty parameter set.
    ///
    /// Returns the AABB that was used for invalidation, or `None` when the
    /// dirty set is empty or points at no valid AABB (in which case the cache
    /// is untouched and the dirty set is cleared to keep bookkeeping in sync).
    pub fn invalidate_chunked_cache(
        &mut self,
        cache: &ChunkedMeshCache,
        bvh: &CompiledSdfBvh,
    ) -> Option<AabbPacked> {
        let aabb = self.dirty_aabb(bvh);
        if let Some(a) = aabb.as_ref() {
            cache.invalidate_region(a.min(), a.max());
        }
        self.mark_clean();
        aabb
    }

    /// Rebuild `bvh` in place from `sdf_node` using `CompiledSdfBvh::try_compile`.
    ///
    /// This is the case B.4 wrapper: a stable API surface over full BVH
    /// recompilation. It always produces AABBs consistent with `sdf_node` —
    /// which is the intended semantics when the caller drives updates
    /// SdfNode-first (mutate a field, then refit).
    ///
    /// **Interaction with `apply` / `apply_all`**: the wrapper reads AABB
    /// source values from `sdf_node`, **not** from `Instruction.params[]`.
    /// If bytecode was updated via `apply`, the caller must independently
    /// synchronise `sdf_node` — this module intentionally does not maintain a
    /// `ParamId → SdfNode` field binding, and doing so would duplicate the
    /// canonical DSL front-end's responsibility.
    ///
    /// True partial refit (bytecode-driven, O(dirty × depth)) is future work
    /// and would remove the SdfNode dependency entirely.
    ///
    /// # Errors
    ///
    /// Forwards any [`CompileError`] returned by
    /// [`CompiledSdfBvh::try_compile`] (unsupported primitive, stack overflow).
    /// On error the original `bvh` is left untouched.
    pub fn refit_bvh(
        &self,
        bvh: &mut CompiledSdfBvh,
        sdf_node: &SdfNode,
    ) -> Result<(), CompileError> {
        let fresh = CompiledSdfBvh::try_compile(sdf_node)?;
        *bvh = fresh;
        Ok(())
    }
}

fn union_of_instruction_aabbs<I: IntoIterator<Item = usize>>(
    indices: I,
    bvh: &CompiledSdfBvh,
) -> Option<AabbPacked> {
    let mut acc: Option<AabbPacked> = None;
    for idx in indices {
        let Some(aabb) = bvh.aabbs.get(idx) else {
            continue;
        };
        if !aabb.is_valid() {
            continue;
        }
        acc = Some(match acc {
            None => *aabb,
            Some(prev) => prev.union(aabb),
        });
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::ChunkedCacheConfig;
    use crate::types::SdfNode;
    use glam::Vec3;

    fn build_simple_scene() -> CompiledSdf {
        // Union(Sphere(1.0), Translate(2, 0, 0)(Box(0.5, 0.5, 0.5)))
        let sphere = SdfNode::sphere(1.0);
        let cube = SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0);
        CompiledSdf::compile(&sphere.union(cube))
    }

    fn build_simple_scene_with_bvh() -> (SdfNode, CompiledSdf, CompiledSdfBvh) {
        let sphere = SdfNode::sphere(1.0);
        let cube = SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0);
        let scene = sphere.union(cube);
        let compiled = CompiledSdf::compile(&scene);
        let bvh = CompiledSdfBvh::compile(&scene);
        (scene, compiled, bvh)
    }

    fn find_first_opcode_index(compiled: &CompiledSdf, opcode: crate::compiled::OpCode) -> usize {
        compiled
            .instructions
            .iter()
            .position(|i| i.opcode == opcode)
            .expect("opcode not found in compiled scene")
    }

    #[test]
    fn empty_index_reports_zero_counts() {
        let index = ParamDependencyIndex::new();
        assert_eq!(index.param_count(), 0);
        assert_eq!(index.binding_count(), 0);
        assert!(!index.is_dirty());
        assert!(index.bindings_of(ParamId::from_raw(0)).is_empty());
    }

    #[test]
    fn bind_single_param_records_one_binding() {
        let compiled = build_simple_scene();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let mut index = ParamDependencyIndex::new();
        let radius_param = ParamId::from_raw(0);
        index
            .bind(radius_param, &compiled, sphere_idx, 0)
            .expect("bind should succeed");
        assert_eq!(index.param_count(), 1);
        assert_eq!(index.binding_count(), 1);
        let slots = index.bindings_of(radius_param);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].instruction_index, sphere_idx);
        assert_eq!(slots[0].param_slot, 0);
    }

    #[test]
    fn bind_shared_param_across_multiple_instructions() {
        let compiled = build_simple_scene();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let box_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Box3d);
        let mut index = ParamDependencyIndex::new();
        let shared = ParamId::from_raw(3);
        index.bind(shared, &compiled, sphere_idx, 0).unwrap();
        index.bind(shared, &compiled, box_idx, 0).unwrap();
        index.bind(shared, &compiled, box_idx, 1).unwrap();
        assert_eq!(index.param_count(), 1);
        assert_eq!(index.binding_count(), 3);
        assert_eq!(index.bindings_of(shared).len(), 3);
    }

    #[test]
    fn bind_rejects_out_of_bounds_instruction() {
        let compiled = build_simple_scene();
        let mut index = ParamDependencyIndex::new();
        let err = index
            .bind(ParamId::from_raw(0), &compiled, 9_999, 0)
            .expect_err("out-of-bounds instruction must fail");
        assert!(matches!(
            err,
            IncrementalError::InstructionOutOfBounds { .. }
        ));
    }

    #[test]
    fn bind_rejects_out_of_range_slot() {
        let compiled = build_simple_scene();
        let mut index = ParamDependencyIndex::new();
        let err = index
            .bind(ParamId::from_raw(0), &compiled, 0, 7)
            .expect_err("slot 7 must fail (valid: 0..7)");
        assert!(matches!(
            err,
            IncrementalError::ParamSlotOutOfRange { slot: 7 }
        ));
    }

    #[test]
    fn apply_writes_value_into_all_bound_slots() {
        let mut compiled = build_simple_scene();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let box_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Box3d);
        let mut index = ParamDependencyIndex::new();
        let pid = ParamId::from_raw(0);
        index.bind(pid, &compiled, sphere_idx, 0).unwrap();
        index.bind(pid, &compiled, box_idx, 0).unwrap();
        index.bind(pid, &compiled, box_idx, 1).unwrap();

        index.apply(&mut compiled, pid, 2.5).unwrap();

        assert_eq!(compiled.instructions[sphere_idx].params[0], 2.5);
        assert_eq!(compiled.instructions[box_idx].params[0], 2.5);
        assert_eq!(compiled.instructions[box_idx].params[1], 2.5);
        assert!(index.is_dirty());
    }

    #[test]
    fn apply_unbound_param_is_silent_no_op() {
        let mut compiled = build_simple_scene();
        let mut index = ParamDependencyIndex::new();
        let unbound = ParamId::from_raw(42);
        index.apply(&mut compiled, unbound, 99.0).unwrap();
        assert!(!index.is_dirty());
    }

    #[test]
    fn apply_all_pushes_solver_values_into_bytecode() {
        let mut compiled = build_simple_scene();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let box_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Box3d);
        let mut index = ParamDependencyIndex::new();
        let radius = ParamId::from_raw(0);
        let width = ParamId::from_raw(1);
        index.bind(radius, &compiled, sphere_idx, 0).unwrap();
        index.bind(width, &compiled, box_idx, 0).unwrap();

        let solver = ConstraintSolver::new(vec![1.75, 0.9]);
        index.apply_all(&mut compiled, &solver).unwrap();

        assert!((compiled.instructions[sphere_idx].params[0] - 1.75).abs() < 1e-6);
        assert!((compiled.instructions[box_idx].params[0] - 0.9).abs() < 1e-6);
        assert_eq!(index.dirty_params().count(), 2);
    }

    #[test]
    fn apply_all_rejects_param_beyond_solver() {
        let mut compiled = build_simple_scene();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let mut index = ParamDependencyIndex::new();
        index
            .bind(ParamId::from_raw(5), &compiled, sphere_idx, 0)
            .unwrap();
        let solver = ConstraintSolver::new(vec![0.0]); // only param 0 addressable
        let err = index
            .apply_all(&mut compiled, &solver)
            .expect_err("param 5 must fail against 1-param solver");
        assert!(matches!(err, IncrementalError::ParamIdOutOfRange { .. }));
    }

    #[test]
    fn dirty_instructions_reflect_recent_apply_calls() {
        let mut compiled = build_simple_scene();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let box_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Box3d);
        let mut index = ParamDependencyIndex::new();
        let radius = ParamId::from_raw(0);
        let box_h = ParamId::from_raw(1);
        index.bind(radius, &compiled, sphere_idx, 0).unwrap();
        index.bind(box_h, &compiled, box_idx, 1).unwrap();

        index.apply(&mut compiled, radius, 2.0).unwrap();
        let dirty: HashSet<usize> = index.dirty_instructions().collect();
        assert_eq!(dirty, HashSet::from([sphere_idx]));

        index.apply(&mut compiled, box_h, 0.75).unwrap();
        let dirty: HashSet<usize> = index.dirty_instructions().collect();
        assert_eq!(dirty, HashSet::from([sphere_idx, box_idx]));

        index.mark_clean();
        assert!(!index.is_dirty());
        assert_eq!(index.dirty_instructions().count(), 0);
    }

    // ── case B: affected_aabb / dirty_aabb / invalidate_chunked_cache ──

    #[test]
    fn affected_aabb_returns_none_for_unbound_param() {
        let (_, _compiled, bvh) = build_simple_scene_with_bvh();
        let index = ParamDependencyIndex::new();
        assert!(index.affected_aabb(ParamId::from_raw(0), &bvh).is_none());
    }

    #[test]
    fn affected_aabb_covers_single_primitive() {
        let (_, compiled, bvh) = build_simple_scene_with_bvh();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let mut index = ParamDependencyIndex::new();
        let radius = ParamId::from_raw(0);
        index.bind(radius, &compiled, sphere_idx, 0).unwrap();

        let aabb = index
            .affected_aabb(radius, &bvh)
            .expect("sphere binding must yield a valid AABB");
        let expected = bvh.aabbs[sphere_idx];
        assert!((aabb.min() - expected.min()).length() < 1e-4);
        assert!((aabb.max() - expected.max()).length() < 1e-4);
    }

    #[test]
    fn affected_aabb_unions_multiple_bindings() {
        let (_, compiled, bvh) = build_simple_scene_with_bvh();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let box_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Box3d);
        let mut index = ParamDependencyIndex::new();
        let shared = ParamId::from_raw(0);
        index.bind(shared, &compiled, sphere_idx, 0).unwrap();
        index.bind(shared, &compiled, box_idx, 0).unwrap();

        let aabb = index.affected_aabb(shared, &bvh).unwrap();
        let expected = bvh.aabbs[sphere_idx].union(&bvh.aabbs[box_idx]);
        assert!((aabb.min() - expected.min()).length() < 1e-4);
        assert!((aabb.max() - expected.max()).length() < 1e-4);
    }

    #[test]
    fn dirty_aabb_is_none_when_no_param_is_dirty() {
        let (_, compiled, bvh) = build_simple_scene_with_bvh();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let mut index = ParamDependencyIndex::new();
        index
            .bind(ParamId::from_raw(0), &compiled, sphere_idx, 0)
            .unwrap();
        assert!(index.dirty_aabb(&bvh).is_none());
    }

    #[test]
    fn dirty_aabb_reflects_apply_calls() {
        let (_, mut compiled, bvh) = build_simple_scene_with_bvh();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let box_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Box3d);
        let mut index = ParamDependencyIndex::new();
        let radius = ParamId::from_raw(0);
        let box_h = ParamId::from_raw(1);
        index.bind(radius, &compiled, sphere_idx, 0).unwrap();
        index.bind(box_h, &compiled, box_idx, 1).unwrap();

        index.apply(&mut compiled, radius, 1.5).unwrap();
        let after_first = index.dirty_aabb(&bvh).unwrap();
        let expected_first = bvh.aabbs[sphere_idx];
        assert!((after_first.min() - expected_first.min()).length() < 1e-4);

        index.apply(&mut compiled, box_h, 0.6).unwrap();
        let after_second = index.dirty_aabb(&bvh).unwrap();
        let expected_second = bvh.aabbs[sphere_idx].union(&bvh.aabbs[box_idx]);
        assert!((after_second.min() - expected_second.min()).length() < 1e-4);
        assert!((after_second.max() - expected_second.max()).length() < 1e-4);
    }

    #[test]
    fn invalidate_chunked_cache_returns_dirty_aabb_and_clears_dirty_set() {
        let (_, mut compiled, bvh) = build_simple_scene_with_bvh();
        let sphere_idx = find_first_opcode_index(&compiled, crate::compiled::OpCode::Sphere);
        let mut index = ParamDependencyIndex::new();
        let radius = ParamId::from_raw(0);
        index.bind(radius, &compiled, sphere_idx, 0).unwrap();
        index.apply(&mut compiled, radius, 1.5).unwrap();
        assert!(index.is_dirty());

        let cache = ChunkedMeshCache::new(ChunkedCacheConfig::default());
        let invalidated = index
            .invalidate_chunked_cache(&cache, &bvh)
            .expect("dirty aabb must be present");
        let expected = bvh.aabbs[sphere_idx];
        assert!((invalidated.min() - expected.min()).length() < 1e-4);
        assert!(!index.is_dirty(), "dirty set must be cleared");
    }

    #[test]
    fn invalidate_chunked_cache_is_noop_when_dirty_set_is_empty() {
        let (_, _compiled, bvh) = build_simple_scene_with_bvh();
        let mut index = ParamDependencyIndex::new();
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig::default());
        let result = index.invalidate_chunked_cache(&cache, &bvh);
        assert!(result.is_none());
        assert!(!index.is_dirty());
    }

    // ── case B.4: SdfNode-based full refit wrapper ──

    #[test]
    fn refit_bvh_produces_same_aabbs_as_fresh_compile() {
        let (scene, _, mut bvh) = build_simple_scene_with_bvh();
        let index = ParamDependencyIndex::new();
        let baseline = CompiledSdfBvh::compile(&scene);
        index
            .refit_bvh(&mut bvh, &scene)
            .expect("refit must succeed on a valid scene");
        assert_eq!(bvh.instructions.len(), baseline.instructions.len());
        assert_eq!(bvh.aabbs.len(), baseline.aabbs.len());
        for (a, b) in bvh.aabbs.iter().zip(baseline.aabbs.iter()) {
            assert!((a.min() - b.min()).length() < 1e-6);
            assert!((a.max() - b.max()).length() < 1e-6);
        }
    }

    #[test]
    fn refit_bvh_reflects_updated_sdf_node_values() {
        let scene_small = SdfNode::sphere(1.0);
        let mut bvh = CompiledSdfBvh::compile(&scene_small);
        let sphere_idx = bvh
            .instructions
            .iter()
            .position(|i| i.opcode == crate::compiled::OpCode::Sphere)
            .unwrap();
        let baseline_max = bvh.aabbs[sphere_idx].max();
        assert!((baseline_max - Vec3::splat(1.0)).length() < 1e-4);

        let scene_large = SdfNode::sphere(3.5);
        let index = ParamDependencyIndex::new();
        index.refit_bvh(&mut bvh, &scene_large).unwrap();
        let after_max = bvh.aabbs[sphere_idx].max();
        assert!((after_max - Vec3::splat(3.5)).length() < 1e-4);
    }

    #[test]
    fn refit_bvh_leaves_bvh_untouched_on_compile_error() {
        // Triangle primitive is intentionally not supported by BVH compile
        // (see validate_for_bvh_compile in src/compiled/eval_bvh.rs). We
        // start from a supported scene, then attempt to refit with an
        // unsupported one and verify the original bvh is unchanged.
        let (scene, _, mut bvh) = build_simple_scene_with_bvh();
        let original_len = bvh.aabbs.len();
        let original_first = bvh.aabbs[0];

        let bad_scene = SdfNode::Triangle {
            point_a: Vec3::new(0.0, 0.0, 0.0),
            point_b: Vec3::new(1.0, 0.0, 0.0),
            point_c: Vec3::new(0.0, 1.0, 0.0),
        };
        let index = ParamDependencyIndex::new();
        let err = index.refit_bvh(&mut bvh, &bad_scene);
        assert!(err.is_err(), "unsupported primitive must error");

        assert_eq!(bvh.aabbs.len(), original_len, "aabbs length unchanged");
        assert!((bvh.aabbs[0].min() - original_first.min()).length() < 1e-6);
        assert!((bvh.aabbs[0].max() - original_first.max()).length() < 1e-6);

        // Silence unused warning
        let _ = scene;
    }
}
