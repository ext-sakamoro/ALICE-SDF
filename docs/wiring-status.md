# ALICE-SDF Wiring Status

_Generated from `scripts/wiring-baseline.txt` and `scripts/wiring_guard.py` (no timestamp: the file changes only when its content does)._

## Status

🟡 **88 baseline items** — Permitted violations, ratchet in place

---

## 📋 Baseline (88 permitted)

Violations explicitly allowed via `scripts/wiring-baseline.txt`.
Must resolve or remove from baseline to reduce ratchet.

### By file

| File | Baseline lines |
|------|----------------|
| `src/compiled/glsl/render_pipeline.rs` | 16 |
| `src/crispy.rs` | 16 |
| `src/compiled/wgsl/gpu_eval.rs` | 13 |
| `src/compiled/glsl/transpiler.rs` | 5 |
| `src/compiled/wgsl/transpiler.rs` | 5 |
| `src/font_bridge.rs` | 5 |
| `src/ffi/registry.rs` | 4 |
| `src/compiled/hlsl/transpiler.rs` | 3 |
| `src/compiled/simd.rs` | 3 |
| `src/ffi/types.rs` | 3 |
| `src/mesh/nanite.rs` | 2 |
| `src/npr/compiled_color.rs` | 2 |
| `src/primitives/mod.rs` | 2 |
| `src/python/helpers.rs` | 2 |
| `src/bin/main.rs` | 1 |
| `src/compiled/transpiler_common.rs` | 1 |
| `src/gi/mod.rs` | 1 |
| `src/io/asdf.rs` | 1 |
| `src/modifiers/surface_roughness.rs` | 1 |
| `src/neural.rs` | 1 |
| `src/volume/export.rs` | 1 |

### Dead Code (10)

```
dead_code src/bin/main.rs 1
dead_code src/compiled/glsl/transpiler.rs 2
dead_code src/compiled/hlsl/transpiler.rs 2
dead_code src/compiled/wgsl/transpiler.rs 2
dead_code src/ffi/registry.rs 3
dead_code src/io/asdf.rs 2
dead_code src/neural.rs 1
dead_code src/npr/compiled_color.rs 1
dead_code src/python/helpers.rs 1
dead_code src/volume/export.rs 1
```

### Unwired Items (78)

```
unwired src/compiled/glsl/render_pipeline.rs::BIOME_SYSTEM
unwired src/compiled/glsl/render_pipeline.rs::DESTRUCTION_SYSTEM
unwired src/compiled/glsl/render_pipeline.rs::DESTRUCTION_UNIFORMS
unwired src/compiled/glsl/render_pipeline.rs::INTERIOR_MAPPING_LIB
unwired src/compiled/glsl/render_pipeline.rs::MICRO_NORMAL_LIB
unwired src/compiled/glsl/render_pipeline.rs::NOISE_LIB
unwired src/compiled/glsl/render_pipeline.rs::NORMAL_AO_SHADOW
unwired src/compiled/glsl/render_pipeline.rs::PBR_BRDF
unwired src/compiled/glsl/render_pipeline.rs::POST_PROCESS
unwired src/compiled/glsl/render_pipeline.rs::SKY_ATMOSPHERE
unwired src/compiled/glsl/render_pipeline.rs::SPECTRAL_LIB
unwired src/compiled/glsl/render_pipeline.rs::UNIFORMS
unwired src/compiled/glsl/render_pipeline.rs::VFX_LIB
unwired src/compiled/glsl/render_pipeline.rs::VOLUMETRIC_LIGHT
unwired src/compiled/glsl/render_pipeline.rs::build_full_shader
unwired src/compiled/glsl/render_pipeline.rs::build_main_function
unwired src/compiled/glsl/transpiler.rs::export_unity_shader_graph
unwired src/compiled/glsl/transpiler.rs::to_fragment_shader
unwired src/compiled/glsl/transpiler.rs::to_fragment_shader_full
unwired src/compiled/glsl/transpiler.rs::to_unity_custom_function
unwired src/compiled/hlsl/transpiler.rs::export_ue5_material_function
unwired src/compiled/hlsl/transpiler.rs::to_ue5_custom_node
unwired src/compiled/simd.rs::max_component
unwired src/compiled/simd.rs::max_zero
unwired src/compiled/simd.rs::min_component
unwired src/compiled/transpiler_common.rs::SHADER_UNSUPPORTED
unwired src/compiled/wgsl/gpu_eval.rs::create_buffer_pool
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_async
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_auto
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_full
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_pooled
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_submit
unwired src/compiled/wgsl/gpu_eval.rs::from_glsl_compute
unwired src/compiled/wgsl/gpu_eval.rs::from_shader_async
unwired src/compiled/wgsl/gpu_eval.rs::from_wgsl_async
unwired src/compiled/wgsl/gpu_eval.rs::new_async
unwired src/compiled/wgsl/gpu_eval.rs::new_dynamic
unwired src/compiled/wgsl/gpu_eval.rs::resolve
unwired src/compiled/wgsl/gpu_eval.rs::wait
unwired src/compiled/wgsl/transpiler.rs::to_compute_shader_with_normals
unwired src/compiled/wgsl/transpiler.rs::to_volume_shader
unwired src/compiled/wgsl/transpiler.rs::transpile_material
unwired src/compiled/wgsl/transpiler.rs::with_workgroup_size
unwired src/crispy.rs::BitMask64::test
unwired src/crispy.rs::BloomFilter::test
unwired src/crispy.rs::EMPTY
unwired src/crispy.rs::FULL
unwired src/crispy.rs::and
unwired src/crispy.rs::branchless_abs
unwired src/crispy.rs::branchless_clamp
unwired src/crispy.rs::branchless_max
unwired src/crispy.rs::branchless_min
unwired src/crispy.rs::fast_recip
unwired src/crispy.rs::fast_recip_vec3
unwired src/crispy.rs::from_items
unwired src/crispy.rs::or
unwired src/crispy.rs::round_half_up_vec3
unwired src/crispy.rs::select_f32
unwired src/crispy.rs::test_hash
unwired src/ffi/registry.rs::clear_all
unwired src/ffi/registry.rs::compiled_count
unwired src/ffi/registry.rs::node_count
unwired src/ffi/types.rs::EvalStats
unwired src/ffi/types.rs::ShaderType
unwired src/ffi/types.rs::SoaBatchConfig
unwired src/font_bridge.rs::FontMetrics
unwired src/font_bridge.rs::char_to_sdf2d
unwired src/font_bridge.rs::font_metrics
unwired src/font_bridge.rs::glyph_to_sdf2d
unwired src/font_bridge.rs::text_to_sdf2d
unwired src/gi/mod.rs::PointLight
unwired src/mesh/nanite.rs::select_clusters
unwired src/mesh/nanite.rs::should_render
unwired src/modifiers/surface_roughness.rs::hash3_xyz
unwired src/npr/compiled_color.rs::fallback_op_count
unwired src/primitives/mod.rs::eval_primitive
unwired src/primitives/mod.rs::eval_primitive_unchecked
unwired src/python/helpers.rs::numpy_to_vec3_fast
```

---

## What is the Wiring Guard?

The wiring guard ensures that all public items in `src/` are actually called from production code:

- **Dead Code Guard**: Verifies that `#[allow(dead_code)]` has a documented reason
- **Unwired Items**: Detects public functions, structs, etc. that are never called (except in tests)
- **Stale Baseline**: Ensures baseline entries are still needed
- **Brace Balance**: Checks syntax integrity

### Resolving Violations

1. **New violations**: Either implement/wire the item, or add to `scripts/wiring-baseline.txt`
2. **Baseline cleanup**: Remove lines from baseline as violations are resolved
3. **Comments**: Add `// ALLOW-DEAD:` or `// ALLOW-UNWIRED:` with reason (12+ chars)

For details: see `scripts/wiring_guard.py`
