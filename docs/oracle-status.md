# ALICE-SDF Oracle Status

_Generated from `tests/*.rs` (no timestamp: the file changes only when its content does)._

## Summary

| Category | Count |
|----------|-------|
| 🟢 Not ignored | 997 |
| 🔴 Red by design | 0 |
| ⏱ Gated (runtime / diagnostic / manual) | 4 |
| ⚪ Pending (bare `#[ignore]`) | 0 |
| **Total** | **1001** |

`Not ignored` means only that the test carries no `#[ignore]`: this report does not run it.
CI's `cargo test` is what says whether it passes.

## ⏱ Gated (4)

Correct tests that are too slow for every push, or that print a measurement table.
Run them with `python3 scripts/run_ignored.py` or `cargo test --release -- --ignored`.

- `metal_default_fast_math_loses_the_sign_of_zero` (test_msl_metal_oracle.rs) — driver observation: Metal default fast math loses the sign of zero
- `the_alembic_library_reads_the_mesh_and_the_identifier_bit_for_bit` (test_alembic_oracle.rs) — needs the Alembic library programs of tests/oracle (the alembic-oracle CI job)
- `the_alembic_library_refuses_a_broken_header` (test_alembic_oracle.rs) — needs the Alembic library programs of tests/oracle (the alembic-oracle CI job)
- `this_crate_writes_what_the_alembic_library_writes_for_the_same_mesh` (test_alembic_oracle.rs) — needs the Alembic library programs of tests/oracle (the alembic-oracle CI job)

## 🟢 Not ignored (997)

Per-file counts (the test names are in `tests/`):

| File | Tests |
|------|-------|
| `test_provenance.rs` | 50 |
| `test_terrain_destruction_oracle.rs` | 28 |
| `test_gi_volume_oracle.rs` | 23 |
| `test_codec_bridge_oracle.rs` | 22 |
| `npr_shader_validate.rs` | 21 |
| `test_live_sdf_oracle.rs` | 21 |
| `test_lod_nanite_meshlet_oracle.rs` | 21 |
| `test_tight_aabb_fold_oracle.rs` | 21 |
| `test_shape_analysis_oracle.rs` | 18 |
| `test_shader_text_provenance.rs` | 17 |
| `test_smooth_ops_oracle.rs` | 17 |
| `test_compiled_bytecode_oracle.rs` | 15 |
| `test_io_round_trip.rs` | 15 |
| `test_compiled_evaluation.rs` | 14 |
| `test_degenerate_input_oracle.rs` | 14 |
| `test_gpu_law_parity.rs` | 14 |
| `test_io_format_oracle.rs` | 13 |
| `test_texture_fit_oracle.rs` | 13 |
| `test_validity_oracle.rs` | 12 |
| `test_binding_oracle.rs` | 11 |
| `test_cache_correctness.rs` | 11 |
| `test_hlsl_blinkscript_parity.rs` | 11 |
| `test_material_oracle.rs` | 11 |
| `test_mesh_collision_fit_oracle.rs` | 11 |
| `test_rendering_pipeline.rs` | 11 |
| `test_tight_aabb_levelset_oracle.rs` | 11 |
| `test_domain_modifier_oracle.rs` | 10 |
| `test_evaluator_opcode_parity.rs` | 10 |
| `test_metric_field_oracle.rs` | 10 |
| `test_svo_api_oracle.rs` | 10 |
| `test_mc_shared_vertex_oracle.rs` | 9 |
| `test_mesh_fidelity.rs` | 9 |
| `test_mesh_query_oracle.rs` | 9 |
| `test_new_transforms.rs` | 9 |
| `test_physics_bridge_determinism.rs` | 9 |
| `test_relaxed_tracing.rs` | 9 |
| `meshopt_reference_vectors.rs` | 8 |
| `surface_oracle.rs` | 8 |
| `test_batch_operations.rs` | 8 |
| `test_destruction_api_oracle.rs` | 8 |
| `test_g8_tree_text.rs` | 8 |
| `test_glsl_export_oracle.rs` | 8 |
| `test_mesh_reorder_oracle.rs` | 8 |
| `test_node_backend_matrix.rs` | 8 |
| `test_npr_analytic.rs` | 8 |
| `test_point_transform_oracle.rs` | 8 |
| `test_tight_aabb_elongate_oracle.rs` | 8 |
| `test_volume_api_oracle.rs` | 8 |
| `test_animation_oracle.rs` | 7 |
| `test_field_fidelity_oracle.rs` | 7 |
| `test_g8_meaning_id.rs` | 7 |
| `test_g8_mesh_import.rs` | 7 |
| `test_gpu_eval_api_oracle.rs` | 7 |
| `test_npr_bytecode_oracle.rs` | 7 |
| `test_primitive_closed_form_oracle.rs` | 7 |
| `test_raycast_oracle.rs` | 7 |
| `test_shader_no_fused_multiply_add.rs` | 7 |
| `test_sim_bridge_oracle.rs` | 7 |
| `test_terrain_api_oracle.rs` | 7 |
| `deep_tree_drop.rs` | 6 |
| `sdf_semantics_id.rs` | 6 |
| `test_autodiff_oracle.rs` | 6 |
| `test_diff_oracle.rs` | 6 |
| `test_hlsl_export_oracle.rs` | 6 |
| `test_jit_dynamic_oracle.rs` | 6 |
| `test_live_sdf_closed_loop.rs` | 6 |
| `test_mesh_cache_model_oracle.rs` | 6 |
| `test_mesh_codec_oracle.rs` | 6 |
| `test_mesh_fit_hull_oracle.rs` | 6 |
| `test_mesh_orientation.rs` | 6 |
| `test_mesh_overdraw_oracle.rs` | 6 |
| `test_mesh_sign_topology.rs` | 6 |
| `test_round_tie_parity.rs` | 6 |
| `test_sdf2d_oracle.rs` | 6 |
| `test_shader_precise_temporaries.rs` | 6 |
| `test_asp_bridge_oracle.rs` | 5 |
| `test_collision_oracle.rs` | 5 |
| `test_g8_example_text.rs` | 5 |
| `test_mesh_extract_uv_oracle.rs` | 5 |
| `test_mesh_quantization_oracle.rs` | 5 |
| `test_meshopt_filter_oracle.rs` | 5 |
| `test_npr_primitives_oracle.rs` | 5 |
| `test_projective_unfused_oracle.rs` | 5 |
| `test_step_export_oracle.rs` | 5 |
| `test_unordered_no_iteration.rs` | 5 |
| `test_bake_mass_oracle.rs` | 4 |
| `test_constraint_oracle.rs` | 4 |
| `test_det_golden.rs` | 4 |
| `test_dual_contouring_invariants.rs` | 4 |
| `test_eval_grid_oracle.rs` | 4 |
| `test_exact_eval_cache.rs` | 4 |
| `test_gi_api_oracle.rs` | 4 |
| `test_hlsl_dxc_compile.rs` | 4 |
| `test_law_minmax_ties.rs` | 4 |
| `test_mesh_cloud_hermite_oracle.rs` | 4 |
| `test_metal_library_from_source_only.rs` | 4 |
| `test_msl_metal_oracle.rs` | 4 |
| `test_neural_mlp_closed_form.rs` | 4 |
| `test_new_modifiers.rs` | 4 |
| `test_sdf_eval_cache_oracle.rs` | 4 |
| `test_soa_oracle.rs` | 4 |
| `test_svo_query_oracle.rs` | 4 |
| `npr_bytecode_wgsl_validate.rs` | 3 |
| `test_bake_stored_forms.rs` | 3 |
| `test_csg_multi_oracle.rs` | 3 |
| `test_det_near_zero_probe.rs` | 3 |
| `test_glsl_glslang_precise.rs` | 3 |
| `test_gpu_noise_parity.rs` | 3 |
| `test_jit_sincos_det.rs` | 3 |
| `test_law_nan_operands.rs` | 3 |
| `test_law_scalar_minmax_ties.rs` | 3 |
| `test_llm_schema_oracle.rs` | 3 |
| `test_minmax_signed_zero_probe.rs` | 3 |
| `test_neural_oracle.rs` | 3 |
| `test_optimize_stats_oracle.rs` | 3 |
| `test_rust_transpiler_oracle.rs` | 3 |
| `float_environment.rs` | 2 |
| `noise_shader_validate.rs` | 2 |
| `test_aabb_unified_oracle.rs` | 2 |
| `test_det_math_clamp_generation.rs` | 2 |
| `test_det_near_zero_parity.rs` | 2 |
| `test_instanced_wgsl_gpu_parity.rs` | 2 |
| `test_interval_predicate_oracle.rs` | 2 |
| `test_interval_soundness.rs` | 2 |
| `test_live_sdf_gpu_parity.rs` | 2 |
| `test_tight_aabb_cover_oracle.rs` | 2 |
| `test_transpiler_naga_validate.rs` | 2 |
| `test_asdf_roundtrip_parity.rs` | 1 |
| `test_det_parity.rs` | 1 |
| `test_gpu_fma_contraction.rs` | 1 |
| `test_metadata_order.rs` | 1 |
| `test_npr_bytecode_gpu_parity.rs` | 1 |
| `test_texture_shader_gpu_parity.rs` | 1 |

---

## Closing an oracle

When an oracle goes green:
1. Remove `#[ignore]` from the test (and the companion test that pins the old behaviour, if the reason says so)
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify
