//! Incremental parameter update support for `CompiledSdf`.
//!
//! This module provides a *binding layer* between [`ParamId`] (parametric
//! variable identifier managed by [`ConstraintSolver`]) and the raw `f32`
//! parameter slots inside [`Instruction`]. It enables incremental workflows
//! such as animation and constraint-driven update, where a single parameter
//! change should only recompute the affected subtree instead of the whole
//! SDF tree.
//!
//! ## Scope (case A: foundation only)
//!
//! - Manual binding registration (`bind`) — the caller declares which
//!   `Instruction.params[slot]` belongs to which `ParamId`.
//! - Value application (`apply` / `apply_all`) — write updated parameter
//!   values into the bytecode in place.
//! - Dirty tracking (`dirty_params` / `dirty_instructions`) — the caller can
//!   iterate affected instruction indices to hook cache invalidation,
//!   e.g. [`ChunkedMeshCache::invalidate_region`].
//!
//! Automatic subtree eval skipping, BVH AABB re-computation, and
//! `MeshCache` auto-invalidation are intentionally out of scope for case A.
//!
//! ## Author
//!
//! Moroya Sakamoto

use crate::compiled::CompiledSdf;
use crate::constraint::{ConstraintSolver, ParamId};
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    fn build_simple_scene() -> CompiledSdf {
        // Union(Sphere(1.0), Translate(2, 0, 0)(Box(0.5, 0.5, 0.5)))
        let sphere = SdfNode::sphere(1.0);
        let cube = SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0);
        CompiledSdf::compile(&sphere.union(cube))
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
}
