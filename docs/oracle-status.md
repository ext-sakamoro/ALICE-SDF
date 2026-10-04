# ALICE-SDF Oracle Status

_Generated from `tests/*.rs` (no timestamp: the file changes only when its content does)._

## Summary

| Category | Count |
|----------|-------|
| 🟢 Not ignored (run by CI) | 362 |
| 🔴 Red by design | 0 |
| ⏱ Gated (runtime / diagnostic / manual) | 0 |
| ⚪ Pending (bare `#[ignore]`) | 0 |
| **Total** | **362** |

`Not ignored` means only that the test carries no `#[ignore]`: this report does not run it.
CI's `cargo test` is what says whether it passes.

## 🟢 Not ignored (362)

Per-file counts (the test names are in `tests/`):

| File | Tests |
|------|-------|
| `test_terrain_destruction_oracle.rs` | 28 |
| `test_gi_volume_oracle.rs` | 23 |
| `npr_shader_validate.rs` | 19 |
| `test_smooth_ops_oracle.rs` | 17 |
| `test_io_round_trip.rs` | 15 |
| `test_compiled_evaluation.rs` | 14 |
| `test_degenerate_input_oracle.rs` | 14 |
| `test_validity_oracle.rs` | 12 |
| `test_gpu_law_parity.rs` | 11 |
| `test_rendering_pipeline.rs` | 11 |
| `test_texture_fit_oracle.rs` | 11 |
| `test_tight_aabb_levelset_oracle.rs` | 11 |
| `test_cache_correctness.rs` | 10 |
| `test_evaluator_opcode_parity.rs` | 10 |
| `test_metric_field_oracle.rs` | 10 |
| `test_binding_oracle.rs` | 9 |
| `test_mesh_fidelity.rs` | 9 |
| `test_new_transforms.rs` | 9 |
| `test_physics_bridge_determinism.rs` | 9 |
| `test_relaxed_tracing.rs` | 9 |
| `meshopt_reference_vectors.rs` | 8 |
| `test_batch_operations.rs` | 8 |
| `test_hlsl_blinkscript_parity.rs` | 8 |
| `test_npr_analytic.rs` | 8 |
| `test_field_fidelity_oracle.rs` | 7 |
| `deep_tree_drop.rs` | 6 |
| `test_mesh_orientation.rs` | 6 |
| `test_mesh_sign_topology.rs` | 6 |
| `test_round_tie_parity.rs` | 6 |
| `test_step_export_oracle.rs` | 5 |
| `test_dual_contouring_invariants.rs` | 4 |
| `test_new_modifiers.rs` | 4 |
| `test_svo_query_oracle.rs` | 4 |
| `npr_bytecode_wgsl_validate.rs` | 3 |
| `test_gpu_noise_parity.rs` | 3 |
| `test_neural_oracle.rs` | 3 |
| `noise_shader_validate.rs` | 2 |
| `test_interval_soundness.rs` | 2 |
| `test_msl_metal_oracle.rs` | 2 |
| `test_transpiler_naga_validate.rs` | 2 |
| `test_asdf_roundtrip_parity.rs` | 1 |
| `test_det_golden.rs` | 1 |
| `test_det_parity.rs` | 1 |
| `test_texture_shader_gpu_parity.rs` | 1 |

---

## How to Contribute

When an oracle goes green:
1. Remove `#[ignore]` from the test (and the companion test that pins the old behaviour, if the reason says so)
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify

For details: [CLAUDE.md](../CLAUDE.md)
